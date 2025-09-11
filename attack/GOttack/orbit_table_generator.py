import os
import numpy as np
import pandas as pd


class OrbitTableGenerator:
    def __init__(self, dataset):
        self.dataset = dataset
        res = os.path.abspath(__file__)  # 获取当前文件绝对路径
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(res)))  # 获取当前文件的上级目录
        self.filepath = base_path + '/dataset/orbit/'

    def generate_orbit_table(self):
        if self.dataset == 'cora':
            return self.generate_orbit_tables_from_sratch()  # acquire the orbit type of each node
        elif self.dataset == 'citeseer':
            return self.generate_orbit_tables_from_sratch()
        elif self.dataset == 'polblogs':
            return self.generate_orbit_tables_from_sratch()
        elif self.dataset == 'BA-SHAPES':
            return self.generate_orbit_tables_from_sratch_other()
        elif self.dataset == 'TREE-CYCLES':
            return self.generate_orbit_tables_from_sratch_other()
        elif self.dataset == 'Loan-Decision':
            return self.generate_orbit_tables_from_sratch_other()
        elif self.dataset == 'ogbn-arxiv':
            return self.generate_orbit_tables_from_sratch_other()
        else:
            raise Exception("Unsupport dataset")

    def generate_orbit_tables_from_sratch(self):
        filename = self.filepath + "dpr_" + self.dataset + '.out'
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Process the lines to create a list of lists
        data_list = [list(map(int, line.split())) for line in lines]

        # Convert the list of lists to a NumPy array

        graphlet_features = np.array(data_list)  # node-level feature
        mylist = []
        for i in range(len(graphlet_features)):
            arr = graphlet_features[i]  # choose a single node
            sorted_indices = np.argsort(arr)[::-1]  # sort in descending order by data index

            #     if sorted_indices[0] < sorted_indices[1]:
            #         if sorted_indices[1] < sorted_indices[2]:
            #             s1 = str(sorted_indices[0]) + str(sorted_indices[1]) + str(sorted_indices[2])
            #         else:
            #             if sorted_indices[0] < sorted_indices[2]:
            #                 s1 = str(sorted_indices[0]) + str(sorted_indices[2]) + str(sorted_indices[1])
            #             else:
            #                 s1 = str(sorted_indices[2]) + str(sorted_indices[0]) + str(sorted_indices[1])
            #     else:
            #         if sorted_indices[0] < sorted_indices[2]:
            #             s1 = str(sorted_indices[1]) + str(sorted_indices[0]) + str(sorted_indices[2])
            #         else:
            #             if sorted_indices[1] < sorted_indices[2]:
            #                 s1 = str(sorted_indices[1]) + str(sorted_indices[2]) + str(sorted_indices[0])
            #             else:
            #                 s1 = str(sorted_indices[2]) + str(sorted_indices[1]) + str(sorted_indices[0])
            #
            #     mylist.append([i, str(sorted_indices[0]), str(sorted_indices[1]), str(sorted_indices[2]), s1])

            if sorted_indices[0] < sorted_indices[1]:
                s1 = str(sorted_indices[0]) + str(sorted_indices[1])
            else:
                s1 = str(sorted_indices[1]) + str(sorted_indices[0])

            mylist.append([i, str(sorted_indices[0]), str(sorted_indices[1]), s1])  # the definition of the orbit type

        my_array = np.array(mylist)
        # values, counts = np.unique(my_array[:,4], return_counts=True)
        # sorted_indices = np.argsort(counts)[::-1]
        # sorted_values = values[sorted_indices]
        # sorted_counts = counts[sorted_indices]
        df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
        return df_2d

    def generate_orbit_tables_from_sratch_other(self):
        if self.dataset == 'BA-SHAPES':
            filename = self.filepath + "BAShapes_orbit_df.csv"
        elif self.dataset == 'TREE-CYCLES':
            filename = self.filepath + "TreeCycle_orbit_df.csv"
        elif self.dataset == 'Loan-Decision':
            filename = self.filepath + "LoanDecision_orbit_df.csv"
        elif self.dataset == 'ogbn-arxiv':
            filename = self.filepath + "arxiv_orbit_df.csv"
        else:
            raise Exception("Unsupport dataset")

        df = pd.read_csv(filename, index_col=0)
        df = df.astype(str)
        return df


def generate_orbit_tables_from_count(data_list, nodes_list):
    # Convert the list of lists to a NumPy array

    graphlet_features = np.array(data_list)
    mylist = []
    for i in range(len(graphlet_features)):
        arr = graphlet_features[i]
        sorted_indices = np.argsort(arr)[::-1]
        if sorted_indices[0] < sorted_indices[1]:
            s1 = str(sorted_indices[0]) + str(sorted_indices[1])
        else:
            s1 = str(sorted_indices[1]) + str(sorted_indices[0])

        mylist.append([nodes_list[i], str(sorted_indices[0]), str(sorted_indices[1]), s1])

    my_array = np.array(mylist)
    df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
    return df_2d
