import torch

ERROR = 10
class Matcher():

    def __init__(self, threshold, buffer_size):
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.object_buffer = [] # Object buffer stores information about tracked objects
        self.object_in_frame = [] # Number of objects in each frame
        self.id = 0 # Current ID for assigning to new objects

    
    def match(self, obeject_embeddings, info_list):
        """
        Match current objects to existing objects in the object buffer or assign new IDs.

        Args:
        - object_embeddings (list): List of embeddings for the current objects
        - info_list (list): List of information about the current objects

        Returns:
        - id_list (list): List of IDs assigned to the current objects
        """
        
        id_list = [-1] * len(obeject_embeddings)
        
        motion_tracklet = [[0] * 2] * len(obeject_embeddings)

        # Record the number of objects in the current frame
        self.object_in_frame.append(len(obeject_embeddings))

        # Matching objects to existing objects in the object buffer
        if self.object_buffer:
            dist_matrix = self.compute_distmatrix(obeject_embeddings)
            
            for _ in range(len(obeject_embeddings)):
                max_dist, row, col = self.get_max(dist_matrix)
                if max_dist == -2 or max_dist < self.threshold:
                    break
                else:
                    matched = 1
                    # x_motion = self.object_buffer[col][3][0]
                    # y_motion = self.object_buffer[col][3][1]
                    # x_offset = info_list[row][0] - self.object_buffer[col][1][0]
                    # y_offset = info_list[row][1] - self.object_buffer[col][1][1]
                    # matched, motion_tracklet[row] = self.rule_1(x_motion, y_motion, x_offset, y_offset)
                    # id_list[row] = self.object_buffer[col][2]
                    
                    if matched == 1:
                        id_list[row] = self.object_buffer[col][2]
                        dist_matrix[row,:] = -2
                        dist_matrix[:,col] = -2
                    else:
                        dist_matrix[row][col] = -2
                        _ -= 1
                

        # Assigning new IDs to unmatched objects
        for i in range(len(obeject_embeddings)):
            if id_list[i] == -1:
                id_list[i] = self.get_id(obeject_embeddings[i])
        
        # Add current objects into the object buffer
        for i in range(len(obeject_embeddings)):
            if id_list[i] == -1:
                id_list[i] = self.id
                self.id += 1
            object_info = [obeject_embeddings[i], info_list[i], id_list[i], motion_tracklet[i]]
            self.object_buffer.append(object_info)

        # Remove old objects from the object buffer if buffer size exceeds the limit
        if len(self.object_in_frame) > self.buffer_size:
            for i in range(self.object_in_frame[0]):
                self.object_buffer.pop(0)
            self.object_in_frame.pop(0)
        
        return id_list

    def get_max(self, dist_matrix):

        """
        Get the maximum value and its corresponding indices from the distance matrix.

        Args:
        - dist_matrix (tensor): Distance matrix between current and existing objects

        Returns:
        - max_dist (float): Maximum distance value
        - row (int): Index of the row containing the maximum value
        - col (int): Index of the column containing the maximum value
        """

        max_dist = dist_matrix.max()
        max_index = dist_matrix.argmax()
        row = (max_index // len(dist_matrix[0])).item() 
        col = (max_index % len(dist_matrix[0])).item()
        return max_dist, row, col

    def compute_distmatrix(self, object_embeddings):

        """
        Compute the cosine similarity distance matrix between current and existing object embeddings.

        Args:
        - object_embeddings (list): Embeddings for current objects

        Returns:
        - dist_matrix (tensor): Distance matrix between current and existing objects
        """

        y_len = len(self.object_buffer)
        x_len= len(object_embeddings)
        dist_matrix = torch.empty((x_len, y_len))
        for i in range(x_len):
            for j in range(y_len):    
                dist_matrix[i][j] = torch.nn.functional.cosine_similarity(object_embeddings[i], self.object_buffer[j][0], dim=0)
        return dist_matrix
    
    def get_id(self, obeject_embeddings):

        """
        Generate and return a new ID for an unmatched object.

        Args:
        - object_embeddings (tensor): Embeddings for the unmatched object

        Returns:
        - new_id (int): New ID for the unmatched object
        """
        
        self.id+=1
        return self.id-1


    def rule_1(self, x_motion, y_motion, x_offset, y_offset):   
        matched = 0
        motion = [0, 0]
        if x_motion == 0 and y_motion == 0:
 
            motion[0] = -1 if x_offset < 0 else 1
            motion[1] = -1 if y_offset < 0 else 1
            matched = 1
        else:
            x_offset -= ERROR * (1 if x_motion < 0 else -1)
            y_offset -= ERROR * (1 if y_motion < 0 else -1)
            if x_offset > 0 and y_offset > 0 and x_motion > 0 and y_motion > 0:
                motion = [1, 1]
                matched = 1
            elif x_offset < 0 and y_offset < 0 and x_motion < 0 and y_motion < 0:
                motion = [-1, -1]
                matched = 1
            elif x_offset > 0 and y_offset < 0 and x_motion > 0 and y_motion < 0:
                motion = [1, -1]
                matched = 1
            elif x_offset < 0 and y_offset > 0 and x_motion < 0 and y_motion > 0:
                motion = [-1, 1]
                matched = 1
        return matched, motion