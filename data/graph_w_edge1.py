import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil, time
from tqdm import tqdm
import os.path as osp

from pandas import read_sql, concat
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.neighbors import kneighbors_graph as knn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from spektral.data import Dataset, Graph


class graph_w_edge1(Dataset):
    """
    Class for a graph dataset with the following edge features:
    distance, time-difference and unit vector for direction
    The node features are normalized with a robust scaler before calculating edge features
    """

    def __init__(self, n_data = None, n_neighbors = 6, **kwargs):
        self.n_data = n_data
        self.n_neighbors = n_neighbors
        super().__init__(**kwargs)

    @property
    def path(self):
        """
        Set the path of the data to be in the processed folder
        """
        cwd = osp.dirname(osp.realpath(__file__))
        path = osp.join(cwd, "processed/graph_w_edge1")
        return path

    def download(self):
        download_start = time.time()
        # Get raw_data
        db_folder = osp.join(osp.dirname(osp.realpath(__file__)), "raw_files")
        db_list   = os.listdir(db_folder)

        # Make output folder
        os.mkdir(self.path )

        # Create data loop 
        downloaded = 0
        seq, sca = None, None
        if self.n_data == None:
            self.n_data = 999999999999
        print("Starting data generation")
        for file in db_list:
            with sqlite3.connect(osp.join(db_folder, file)) as con:
                print(f"Reading {file}")
                distinct = read_sql(f"select distinct event_no from sequential", con).event_no
                if len(distinct) <= self.n_data - downloaded:
                    limit = distinct.max()
                else:
                    limit = distinct.sort_values()[self.n_data - downloaded]
                if type(seq) == None:
                    seq     = read_sql(f"select * from sequential where event_no < {limit};", con)
                    sca     = read_sql(f"select * from scalar where event_no < {limit};", con)
                else:
                    seq     = concat([seq, read_sql(f"select * from sequential where event_no < {limit};", con)])
                    sca     = concat([sca, read_sql(f"select * from scalar where event_no < {limit};", con)])
<<<<<<< HEAD
            downloaded += len(seq.event_no.unique())
=======
            downloaded = len(seq.event_no.unique())
>>>>>>> dbb8ad9bd1749f2ac8d9aa88b752629e1547efdb
            if downloaded >= self.n_data:
                print(f"Succesfully loaded data for {downloaded} graphs")
                break
            if file == db_list[-1]:
                print(f"All raw  data loaded: {downloaded} graphs")
        
        # Set attributes
        node_cols = ["dom_x", "dom_y", "dom_z", "dom_charge", "dom_time"]
        event_no  = seq.event_no.unique()[sca.event_no.isin(sca.event_no)]
        seq       = seq[seq.event_no.isin(event_no)]
        sca       = sca[sca.event_no.isin(event_no)]
        targets   = ["true_primary_energy"]

        idx_nodes = np.array(seq.event_no)
        idx_targ  = np.array(sca.event_no)

        nodes_arr = np.array(seq.loc[:, node_cols])
        targ_arr  = np.array(sca.loc[:, targets])

        nodes_arr  = RobustScaler().fit_transform(nodes_arr)
        targ_arr  = RobustScaler().fit_transform(targ_arr)

        # Find cuts for node_arr
        ids, start, count = np.unique(idx_nodes, return_index = True, return_counts = True)

        graph_list = []
        # Just add a range for the y array, shou
        for i, id, s, c in tqdm(zip(range(len(ids)), ids, start, count), total = len(ids)):
            x = nodes_arr[s: s + c + 1 , :]
            y = targ_arr[i]

            a, e = calculate_edge_attributes(x, n_neighbors = self.n_neighbors)

            graph_list.append((x, a, e, y))
            if (i % 10000 == 0 and i > 0) or i == len(ids) - 1:
              graph_list =np.array(graph_list, dtype = object)
              np.savez(osp.join(self.path, f"graphs{len(os.listdir(self.path))}"), graph_list)
              graph_list = []


        # print("Saving file")
        # start_time = time.time()
        # graph_list =np.array(graph_list, dtype = object)
        # for i in tqdm(range(0, len(graph_list), 10000)):
        #     save_now = graph_list[i:i+10000]
        #     np.savez(osp.join(self.path, f"graphs{i}"), save_now)
        # print(f"Data saved in {time.time() - start_time:.1f} seconds")
        print(f"Total time to create dataset: {time.time() - download_start:.1f} seconds")

    def read(self):
        print("Loading data to memory")
        output = []
        for file in tqdm(os.listdir(self.path), position = 0, leave = True):
            graphs = np.load(osp.join(self.path, file), allow_pickle = True)
            for g in graphs["arr_0"]:
                output.append(Graph(*g))
        return output
        



def calculate_edge_attributes(x, n_neighbors):
    pos, charge, time = x[:, :3], x[:, 3], x[:, 4]
    a = knn(x[:, :3], n_neighbors)

    index_i, index_j = np.repeat(np.arange(a.shape[0]), n_neighbors), a.indices

    dists = np.linalg.norm(pos[index_i] - pos[index_j], axis = 1)
    vects = normalize(pos[index_i] - pos[index_j])
    dts = time[index_j] - time[index_i]

    e = np.vstack(np.vstack([dists, dts, vects.T])).T

    return a, e



if __name__ == "__main__":
    print("Removing current data folder")
    
    path = osp.dirname(osp.realpath(__file__))
    if not "processed" in os.listdir(path):
        os.mkdir(osp.join(path, "processed"))
    if not "raw_files" in os.listdir(path):
        os.mkdir(osp.join(path, "processed"))
        print("Folder created for raw files, please add some before continuing")
        sys.exit()

    if os.path.isdir(osp.join(path, "processed", "graph_w_edge1")):
        shutil.rmtree(osp.join(path, "processed", "graph_w_edge1"))
    if len(sys.argv) == 2:
        n_data = int(sys.argv[1])
        print(f"Preparing dataset with {n_data} graphs")
    else:
        n_data = None
        print("Preparing dataset with all availible raw data")

    # Preparing data 
    dataset = graph_w_edge1(n_data)
