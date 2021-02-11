import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil, time
from tqdm import tqdm
import os.path as osp
from scipy.sparse import csr_matrix

from pandas import read_sql, concat
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.neighbors import kneighbors_graph as knn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from spektral.data import Dataset, Graph
import tensorflow as tf



features = ["dom_x", "dom_y", "dom_z", "time", "charge_log10"]
targets  = ["energy_log10", "position_x", "position_y", "position_z", "azimuth", "zenith", "pid"]

muon = True

class graph_w_edge2(Dataset):
    """
    Class for a graph dataset with the following edge features:
    distance, time-difference and unit vector for direction
    The node features are normalized with a robust scaler before calculating edge features
    """

    def __init__(self, n_data = None, skip = 0, n_neighbors = 6, train_val_test_split = [0.8, 0.1, 0.1], **kwargs):
        self.n_data = n_data
        self.skip   = skip
        self.n_neighbors = n_neighbors
        if sum(train_val_test_split) != 1:
            sys.exit("Train, test, val ratios must add up to 1")
        self.train_size, self.val_size, self.test_split = train_val_test_split
        self.seed = 25

        super().__init__(**kwargs)

    @property
    def path(self):
        """
        Set the path of the data to be in the processed folder
        """
        cwd = osp.dirname(osp.realpath(__file__))
        path = osp.join(cwd, "processed/graph_w_edge2_angles")
        return path


    def download(self):
        download_start = time.time()
        # Get raw_data
        db_folder = osp.join(osp.dirname(osp.realpath(__file__)), "db_files")
        db_file   = osp.join(db_folder, "rasmus_classification_muon_3neutrino_3mio.db")

        # Make output folder
        os.mkdir(self.path)

        print("Connecting to db-file")
        with sqlite3.connect(db_file) as conn:
            # Find indices to cut after
            try:
                if muon:
                    start_id = conn.execute(f"select distinct event_no from truth where pid = 13 limit 1 offset {self.skip}").fetchall()[0][0]
                    stop_id  = conn.execute(f"select distinct event_no from truth where pid = 13 limit 1 offset {self.skip + self.n_data}").fetchall()[0][0]
                else:
                    start_id = conn.execute(f"select distinct event_no from truth limit 1 offset {self.skip}").fetchall()[0][0]
                    stop_id  = conn.execute(f"select distinct event_no from truth limit 1 offset {self.skip + self.n_data}").fetchall()[0][0]
            except:
                start_id = 0
                stop_id  = 999999999

            # SQL queries format
            feature_call = ", ".join(features)
            target_call  = ", ".join(targets)

            # Load data from db-file
            print("Reading files")
            df_event = read_sql(f"select event_no       from features where event_no >= {start_id} and event_no < {stop_id} and SRTInIcePulses = 1", conn)
            df_feat  = read_sql(f"select {feature_call} from features where event_no >= {start_id} and event_no < {stop_id} and SRTInIcePulses = 1", conn)
            df_targ  = read_sql(f"select {target_call } from truth    where event_no >= {start_id} and event_no < {stop_id}", conn)
            
            transformers = pickle.load(open(osp.join(db_folder, "transformers.pkl"), 'rb'))
            trans_x      = transformers['features']
            trans_y      = transformers['truth']


            for col in ["dom_x", "dom_y", "dom_z"]:
                df_feat[col] = trans_x[col].inverse_transform(np.array(df_feat[col]).reshape(1, -1)).T

            for col in ["energy_log10", "position_x", "position_y", "position_z", "azimuth", "zenith"]:
                df_targ[col] = trans_y[col].inverse_transform(np.array(df_targ[col]).reshape(1, -1)).T
            
            

            # Cut indices
            print("Splitting data to events")
            idx_list    = np.array(df_event)
            x_not_split = np.array(df_feat)

            _, idx, counts = np.unique(idx_list.flatten(), return_index = True, return_counts = True) 
            xs          = np.split(x_not_split, np.cumsum(counts)[:-1])

            ys          = np.array(df_targ)



            # Generate adjacency matrices
            print("Generating adjacency matrices")
            graph_list = []
            for x, y in tqdm(zip(xs, ys), total = len(xs)):
                try:
                    a = knn(x[:, :3], self.n_neighbors)
                except:
                    a = csr_matrix(np.ones(shape = (x.shape[0], x.shape[0])) - np.eye(x.shape[0]))
                
                graph_list.append(Graph(x = x, a = a, y = y))

            graph_list = np.array(graph_list, dtype = object)


            print("Saving dataset")
            pickle.dump(graph_list, open(osp.join(self.path, "data.dat"), 'wb'))

        
    def read(self):
        print("Loading data to memory")
        data   = pickle.load(open(osp.join(self.path, "data.dat"), 'rb'))


        np.random.seed(self.seed)
        idxs = np.random.permutation(len(data))
        train_split = int(self.train_size * len(data))
        val_split   = int(self.val_size * len(data)) + train_split

        idx_tr, idx_val, idx_test  = np.split(idxs, [train_split, val_split])
        self.index_lists = [idx_tr, idx_val, idx_test]

        return data
        


if __name__ == "__main__":
    print("Removing current data folder")
    
    path = osp.dirname(osp.realpath(__file__))
    if not "processed" in os.listdir(path):
        os.mkdir(osp.join(path, "processed"))
    if not "raw_files" in os.listdir(path):
        os.mkdir(osp.join(path, "processed"))
        print("Folder created for raw files, please add some before continuing")
        sys.exit()

    if os.path.isdir(osp.join(path, "processed", "graph_w_edge2_angles")):
        shutil.rmtree(osp.join(path, "processed", "graph_w_edge2_angles"))
    if len(sys.argv) == 2:
        n_data = int(sys.argv[1])
        print(f"Preparing dataset with {n_data} graphs")
    else:
        n_data = None
        print("Preparing dataset with all availible raw data")

    # Preparing data 
    dataset = graph_w_edge2(n_data = n_data, n_neighbors = 8)
