#### Standard imports
import os
from ExergyUtilities import util_spyder

#### Logging (broken for Spyder, so using a wrapper)
#import logging
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
#logging.basicConfig(format=FORMAT,level=logging.DEBUG)
#logging.debug("Test")

logging = util_spyder.SpyderLog(FORMAT)
logging.info("Started logging")

#### External imports - reimported for code completion! 
print_imports()

#%% ===========================================================================
#  Data source and paths
# =============================================================================
PATH_DATA_ROOT = r"/home/batman/Dropbox/DATA/KaggleDays Reddit"
path_data = os.path.join(PATH_DATA_ROOT, r"")
assert os.path.exists(path_data), path_data

#%% ===========================================================================
#  Load Train and Test data
# =============================================================================
logging.info(f"Load train and test")
data_train = pd.read_csv(os.path.join(path_data, "train.zip"),delimiter='\t',compression='zip',index_col='id')
logging.info(f"Loaded {len(data_train)} train rows")

#data_test = pd.read_csv(os.path.join(path_data, "test.csv"),delimiter='\t', )
data_test = pd.read_csv(os.path.join(path_data, "test.zip"),delimiter='\t',compression='zip',index_col='id')
logging.info(f"Loaded {len(data_test)} test rows")

#%% SUBSET; 2 subreddits

# Select a list of categories
selection = ['movies','gaming']

# Empty masks
this_train_filter = np.full(data_train.shape[0],False)
this_test_filter = np.full(data_test.shape[0],False)
for this_category in selection:
    # Concantenate for train
    selected_train_rows = data_train['subreddit'].values == this_category
    this_train_filter = this_train_filter | selected_train_rows
    # Concantenate for test
    selected_test_rows = data_test['subreddit'].values == this_category
    this_test_filter = this_test_filter | selected_test_rows
    logging.debug(f"({np.sum(selected_train_rows)} {np.sum(selected_test_rows)}) (train, test) matches for category: {this_category}")

frac_tr = np.sum(this_train_filter)/len(data_train)*100
frac_te = np.sum(this_test_filter)/len(data_test)*100

logging.info(f"{np.sum(this_train_filter)} train records selected ({frac_tr:{0}.{2}}%) -> Xy_tr")
logging.info(f"{np.sum(this_test_filter)} test records selected ({frac_te:{0}.{2}}%) -> X_te")

Xy_tr   = data_train[this_train_filter] 
X_te    = data_test[this_test_filter]   

#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
        "data_test",
        "data_train",
        "selected_train_rows",
        "this_train_filter",
        "selected_test_rows",
        "this_test_filter",
        "selection",
        "this_category",
        "path_data",
        "frac_te",
        "frac_te",
        "",
        "",
        #"In",
        #"Out",
        
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars
#%%
#
#keep_vars =[
#        "keep_vars",
#        "Xy_tr",
#        "X_te",
#        "In",
#        "Out",
#        
#        ]
#for name in dir():
#    print(name, end=" ")
#    if name.startswith('__'):
#        print("keep")
#        pass
#    if name.startswith('_'):
#        print("keep")
#        pass    
#    elif name in keep_vars:
#        print("keep")
#        pass
#    else:
#        print("del")
#        del globals()[name]
#    
#raise







