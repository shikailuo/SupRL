import os
import numpy as np
import random
import sys
import operator
import pickle

class ReplayBuffer:
    def __init__(self, size, online=True, 
                 persistent_directory=None,
                 episode_counts_to_save=100, 
                 sample_steps_to_refresh=500):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        online: bool
          whether in online or offline mode.
        persistent_directory: str
          Directory where logged data is saved or to be saved
        episode_counts_to_save: int
          dump replay buffer to logged data every `episode_counts_to_save` episode,
          which name is `trajs_{cur_logged_buffer_index}.pkl`
          NOTE: we need to know when an episode is begin and when it is finished(even if it is not terminated)
          and we need to make sure trajectories in logged data is not duplicated!
        sample_steps_to_refresh: int
          would refresh buffer from logged data when sampled batch counts exceed this threshold.
          NOTE: refresh would from 0 to `max_logged_buffer_index` with format like `logged_0.rb` 
          under `persistent_directory` in cycle.
        """
        self._storage = []
        self._maxsize = size # will ignore this param if offline
        self._count = 0 # len(self._storage)
        self._next_idx = 0
        
        self._online = online
        self._save_to_disk = False # no trajs saved by default

        self._persistent_directory = persistent_directory
        if not self._online: # persistent_directory must exist in OFFLINE mode!
            assert self._persistent_directory is not None, \
                   'Please pass directory where logged trajs are saved!'

        if self._persistent_directory is not None: # online to save, offline to load
            assert os.path.isdir(self._persistent_directory), \
                   'Please make sure {} exists!'.format(self._persistent_directory)
            
            self._cur_file_index = 0 # files array index in OFFLINE mode, file name index in ONLINE mode
            
            if self._online:
                self._save_to_disk = True
                self._logged_file_prefix = 'trajs_{}.pkl'             
                # trajectory storage is a list of list
                self._trajectory_storage = []
                # trajectory is a list of transition
                self._cur_trajectory = []
                # every 'episode_counts_to_save' episodes to save
                self._episode_counts_to_save = episode_counts_to_save
                print('Save buffer every {} episodes!'.format(self._episode_counts_to_save))
                # every time learn online start from 0
                # and advance with `add` when passed in episode index change
                # and used to control save when every `episode_counts_to_save`
                self._cur_episode_counts = 0
                # episode id is passed to detect begin & end of episode
                self._cur_episode_id = 0
            else:
                self._save_to_disk = False
                self._files = os.listdir(self._persistent_directory)
                self._files = sorted([file for file in self._files if file.endswith('.pkl')])
                self._num_files = len(self._files)
                # load offline data
                self.load()
                self._sample_steps_to_refresh = sample_steps_to_refresh
                print('Refresh buffer every {} sampling!'.format(self._sample_steps_to_refresh))
                self._cur_sample_steps = 0

    def save(self, replace=True):
        save_path = self._save_load_path()
        if os.path.exists(save_path) and not replace:
            print('Save path: {} already exists, would not save!'.format(save_path))
            return
        
        with open(save_path, 'wb') as f:
            pickle.dump(self._trajectory_storage, f)
        # advance logged index cursor
        self._cur_file_index += 1
        # empty trajectory storage
        self._trajectory_storage = []
        # empty cur trajectory memory
        self._cur_trajectory = []
        print('Saved trajectories to save path: {}!'.format(save_path))

    def load(self):
        """
        Load trajectories into `self._trajectory_storage`
        and flatten transitions into `self._storage`
        """
        load_path = self._save_load_path()
        if not os.path.exists(load_path):
            print('load path: {} does not exist, no data loaded, please check your path!'.format(load_path))
            return
        with open(load_path, 'rb') as f:
            # actually this is not that need...
            self._trajectory_storage = pickle.load(f)
        # flatten trajectories into transitions
        self._storage = [transition for trajectory in self._trajectory_storage for transition in trajectory]
        # update related params
        self._count = len(self._storage)
        self._maxsize = self._count
        self._cur_file_index = (self._cur_file_index + 1) % self._num_files
        print('Loaded trajectories from load path: {}!'.format(load_path))

    def add(self, obs_t, action, reward, obs_tp1, done, episode_id, weight=None):
        if not self._online:
            print('Could not add data to buffer in OFFLINE mode!')
            return
        
        if self._save_to_disk: # whether to save to disk
            if self._cur_episode_id != episode_id:
                self._cur_episode_id = episode_id
                self._trajectory_storage.append(self._cur_trajectory)
                self._cur_trajectory = []
                self._cur_episode_counts += 1
                if self._cur_episode_counts % self._episode_counts_to_save == 0:
                    self.save()
                
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= self._count: # not full yet
            self._storage.append(data)
            self._count += 1
        else: # full sized buffer
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
        if self._save_to_disk:
            # add current transition to current trajectory memory
            self._cur_trajectory.append(data)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
          batch of observations
        act_batch: np.array
          batch of actions executed given obs_batch
        rew_batch: np.array
          rewards received as results of executing act_batch
        next_obs_batch: np.array
          next set of observations seen after executing act_batch
        done_mask: np.array
          done_mask[i] = 1 if executing act_batch[i] resulted in
          the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, self._count - 1) for _ in range(batch_size)]
        
        encoded_sample = self._encode_sample(idxes)
        
        if not self._online:
            self._cur_sample_steps += 1
            if self._cur_sample_steps % self._sample_steps_to_refresh == 0:
                self.load()
        
        return encoded_sample
        
    def save_class(self, save_dir, save_name):
        """
        Save with pickle, whole class
        """
        save_path = os.path.join(save_dir, save_name)
        if os.path.exists(save_path):
            print('Save path {} already exists, would not save !'.format(save_path))
            return

        if not os.path.exists(save_dir):
            print('Save directory {} is not existed, create it !'.format(save_path))
            os.mkdir(save_dir)

        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print('Saved whole replay buffer to save path: {} !'.format(save_path))
    
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (np.array(obses_t), np.array(actions), np.array(rewards),
                np.array(obses_tp1), np.array(dones))

    def __len__(self):
        return self._count
    
    def _save_load_path(self):
        if self._online:
            file_name = self._logged_file_prefix.format(self._cur_file_index) 
        else:
            file_name = self._files[self._cur_file_index]
        path = os.path.join(self._persistent_directory, file_name)
        return path
    
    def update(self, trajs):
        self._storage = [transition for traj in trajs for transition in traj]
        # update related params
        self._count = len(self._storage)
        self._maxsize = self._count


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha=0.6, beta=0.4, 
                 online=True, persistent_directory=None,
                 episode_counts_to_save=100, 
                 sample_steps_to_refresh=500,
                 debug=False):
        """
        Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        alpha: float
          how much prioritization is used
          (0 - no prioritization, 1 - full prioritization)
        beta: float
          To what degree to use importance weights
          (0 - no corrections, 1 - full correction)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size=size, 
                              online=online,
                              persistent_directory=persistent_directory, 
                              episode_counts_to_save=episode_counts_to_save,                                                     
                              sample_steps_to_refresh=sample_steps_to_refresh)

        assert alpha > 0 and beta >= 0.0
        self._alpha = alpha
        self._beta = beta

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        
    def add(self, obs_t, action, reward, obs_tp1, done, episode_id, weight=None):
        """See ReplayBuffer.store_effect"""

        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).add(obs_t, action, reward,
                                                 obs_tp1, done, episode_id, weight)
        if weight is None:
            weight = self._max_priority
        self._it_sum[idx] = weight**self._alpha
        self._it_min[idx] = weight**self._alpha
        
    def load(self):
        """
        Load trajectories into `self._trajectory_storage`
        and flatten transitions into `self._storage`
        """
        super(PrioritizedReplayBuffer, self).load()
        it_capacity = 1
        while it_capacity < self._count:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, self._count)
            idx = self._it_sum.find_prefixsum_idx(mass)
            if idx >= self._count:
                idx = random.randint(0, self._count - 1) 
            res.append(idx)
        return res

    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
          How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
          batch of observations
        act_batch: np.array
          batch of actions executed given obs_batch
        rew_batch: np.array
          rewards received as results of executing act_batch
        next_obs_batch: np.array
          next set of observations seen after executing act_batch
        done_mask: np.array
          done_mask[i] = 1 if executing act_batch[i] resulted in
          the end of an episode and 0 otherwise.
        weights: np.array
          Array of shape (batch_size,) and dtype np.float32
          denoting importance weight of each sampled transition
        idxes: np.array
          Array of shape (batch_size,) and dtype np.int32
          idexes in buffer of sampled experiences
        """

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._count)**(-self._beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self._count)**(-self._beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        update = True
        
        if not self._online:
            self._cur_sample_steps += 1
            if self._cur_sample_steps % self._sample_steps_to_refresh == 0:
                self.load()
                update = False
                
        return tuple(list(encoded_sample) + [weights, idxes, update])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
          List of idxes of sampled transitions
        priorities: [float]
          List of updated priorities corresponding to
          transitions at the sampled idxes denoted by
          variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self._count
            delta = priority**self._alpha - self._it_sum[idx]
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha

            self._max_priority = max(self._max_priority, priority)
    
############################################################################################
class SegmentTree:
    """
    A Segment Tree data structure.
    https://en.wikipedia.org/wiki/Segment_tree
    Can be used as regular array, but with two important differences:
      a) Setting an item's value is slightly slower. It is O(lg capacity),
         instead of O(1).
      b) Offers efficient `reduce` operation which reduces the tree's values
         over some specified contiguous subsequence of items in the array.
         Operation could be e.g. min/max/sum.
    The data is stored in a list, where the length is 2 * capacity.
    The second half of the list stores the actual values for each index, so if
    capacity=8, values are stored at indices 8 to 15. The first half of the
    array contains the reduced-values of the different (binary divided)
    segments, e.g. (capacity=4):
    0=not used
    1=reduced-value over all elements (array indices 4 to 7).
    2=reduced-value over array indices (4 and 5).
    3=reduced-value over array indices (6 and 7).
    4-7: values of the tree.
    NOTE that the values of the tree are accessed by indices starting at 0, so
    `tree[0]` accesses `internal_array[4]` in the above example.
    """
    def __init__(self, capacity, operation, neutral_element=None):
        """
        Initializes a Segment Tree object.
        Args:
            capacity (int): Total size of the array - must be a power of two.
            operation (operation): Lambda obj, obj -> obj
                The operation for combining elements (eg. sum, max).
                Must be a mathematical group together with the set of
                possible values for array elements.
            neutral_element (Optional[obj]): The neutral element for
                `operation`. Use None for automatically finding a value:
                max: float("-inf"), min: float("inf"), sum: 0.0.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "Capacity must be positive and a power of 2!"
        self.capacity = capacity
        if neutral_element is None:
            neutral_element = 0.0 if operation is operator.add else \
                float("-inf") if operation is max else float("inf")
        self.neutral_element = neutral_element
        self.value = [self.neutral_element for _ in range(2 * capacity)]
        self.operation = operation

    def reduce(self, start=0, end=None):
        """
        Applies `self.operation` to subsequence of our values.
        Subsequence is contiguous, includes `start` and excludes `end`.
          self.operation(
              arr[start], operation(arr[start+1], operation(... arr[end])))
        Args:
            start (int): Start index to apply reduction to.
            end (Optional[int]): End index to apply reduction to (excluded).
        Returns:
            any: The result of reducing self.operation over the specified
                range of `self._value` elements.
        """
        if end is None:
            end = self.capacity
        elif end < 0:
            end += self.capacity

        # Init result with neutral element.
        result = self.neutral_element
        # Map start/end to our actual index space (second half of array).
        start += self.capacity
        end += self.capacity

        # Example:
        # internal-array (first half=sums, second half=actual values):
        # 0 1 2 3 | 4 5 6 7
        # - 6 1 5 | 1 0 2 3

        # tree.sum(0, 3) = 3
        # internally: start=4, end=7 -> sum values 1 0 2 = 3.

        # Iterate over tree starting in the actual-values (second half)
        # section.
        # 1) start=4 is even -> do nothing.
        # 2) end=7 is odd -> end-- -> end=6 -> add value to result: result=2
        # 3) int-divide start and end by 2: start=2, end=3
        # 4) start still smaller end -> iterate once more.
        # 5) start=2 is even -> do nothing.
        # 6) end=3 is odd -> end-- -> end=2 -> add value to result: result=1
        #    NOTE: This adds the sum of indices 4 and 5 to the result.

        # Iterate as long as start != end.
        while start < end:

            # If start is odd: Add its value to result and move start to
            # next even value.
            if start & 1:
                result = self.operation(result, self.value[start])
                start += 1

            # If end is odd: Move end to previous even value, then add its
            # value to result. NOTE: This takes care of excluding `end` in any
            # situation.
            if end & 1:
                end -= 1
                result = self.operation(result, self.value[end])

            # Divide both start and end by 2 to make them "jump" into the
            # next upper level reduce-index space.
            start //= 2
            end //= 2

            # Then repeat till start == end.

        return result

    def __setitem__(self, idx, val):
        """
        Inserts/overwrites a value in/into the tree.
        Args:
            idx (int): The index to insert to. Must be in [0, `self.capacity`[
            val (float): The value to insert.
        """
        assert 0 <= idx < self.capacity

        # Index of the leaf to insert into (always insert in "second half"
        # of the tree, the first half is reserved for already calculated
        # reduction-values).
        idx += self.capacity
        self.value[idx] = val

        # Recalculate all affected reduction values (in "first half" of tree).
        idx = idx >> 1  # Divide by 2 (faster than division).
        while idx >= 1:
            update_idx = 2 * idx  # calculate only once
            # Update the reduction value at the correct "first half" idx.
            self.value[idx] = self.operation(self.value[update_idx],
                                             self.value[update_idx + 1])
            idx = idx >> 1  # Divide by 2 (faster than division).

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        return self.value[idx + self.capacity]

class SumSegmentTree(SegmentTree):
    """A SegmentTree with the reduction `operation`=operator.add."""

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add)

    def sum(self, start=0, end=None):
        """Returns the sum over a sub-segment of the tree."""
        return self.reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Finds highest i, for which: sum(arr[0]+..+arr[i]) <= prefixsum.
        Args:
            prefixsum (float): `prefixsum` upper bound in above constraint.
        Returns:
            int: Largest possible index (i) satisfying above constraint.
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        # Global sum node.
        idx = 1

        # While non-leaf (first half of tree).
        while idx < self.capacity:
            update_idx = 2 * idx
            if self.value[update_idx] > prefixsum:
                idx = update_idx
            else:
                prefixsum -= self.value[update_idx]
                idx = update_idx + 1
        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min)

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""
        return self.reduce(start, end)
