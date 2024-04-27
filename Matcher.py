import torch

class Matcher():

    def __init__(self, threshold, buffer_size):
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.object_buffer = [] #object buffer = [[embedding, info, id, motion],[embedding, info, id, motion]]
        self.object_in_frame = []
        self.id = 0

    def match(self, obeject_embeddings, info_list):

        
        id_list = [-1] * len(obeject_embeddings)

        # record num of object in the frame
        self.object_in_frame.append(len(obeject_embeddings))

        #matching
        if self.object_buffer:
            dist_matrix = self.compute_distmatrix(obeject_embeddings)
            
            for _ in range(len(obeject_embeddings)):
                max_dist, row, col = self.get_max(dist_matrix)
                if max_dist == -2 or max_dist < self.threshold:
                    break
                else:
                    id_list[row] = self.object_buffer[col][2]

                dist_matrix[row,:] = -2
                dist_matrix[:,col] = -2

        #ask gallery
        for i in range(len(obeject_embeddings)):
            if id_list[i] == -1:
                id_list[i] = self.get_id(obeject_embeddings[i])
        
        #add current objects into buffer
        for i in range(len(obeject_embeddings)):
            if id_list[i] == -1:
                id_list[i] = self.id
                self.id += 1
            object_info = [obeject_embeddings[i], info_list[i], id_list[i]]
            self.object_buffer.append(object_info)

    
        if len(self.object_in_frame) > self.buffer_size:
            for i in range(self.object_in_frame[0]):
                self.object_buffer.pop(0)
            self.object_in_frame.pop(0)
        
        return id_list

    def get_max(self, dist_matrix):
        max_dist = dist_matrix.max()
        max_index = dist_matrix.argmax()
        row = (max_index // len(dist_matrix[0])).item() 
        col = (max_index % len(dist_matrix[0])).item()
        return max_dist, row, col

    def compute_distmatrix(self, object_embeddings):
        y_len = len(self.object_buffer)
        x_len= len(object_embeddings)
        dist_matrix = torch.empty((x_len, y_len))
        for i in range(x_len):
            for j in range(y_len):    
                dist_matrix[i][j] = torch.nn.functional.cosine_similarity(object_embeddings[i], self.object_buffer[j][0], dim=0)
        return dist_matrix
    
    def get_id(self, obeject_embeddings):
        self.id+=1
        return self.id-1