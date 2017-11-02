import numpy as np
import xlrd
import theano

def load_data(filename):

    # load data post preprocess1
    simulation_results = xlrd.open_workbook(filename)
    worksheet = simulation_results.sheet_by_index(0)
    
    batchdata = np.asarray(worksheet.col_values(0)).reshape((-1,1))
    
    num_col = worksheet.row_len(0)
    for i in range(1,num_col):
        batchdata = np.concatenate((batchdata, 
                                    np.asarray(worksheet.col_values(i)).reshape((-1,1))), axis=1)
    
    data_max =  batchdata.max(axis=0)
    data_min =  batchdata.min(axis=0)
    
    #batchdata = ((batchdata - data_min) / (data_max - data_min))*2-1.0
    
    data_mean = batchdata.mean(axis=0)
    data_std = batchdata.std(axis=0)
    
    
    batchdata = (batchdata - data_mean) / data_std
    
    # get sequence lengths
    seqlen = batchdata.shape[0]
    
    # put data into shared memory
    shared_x = theano.shared(np.asarray(batchdata, dtype=theano.config.floatX))

    return shared_x, seqlen, data_min, data_max, data_mean, data_std

if __name__ == "__main__":
    batchdata, seqlen, data_min, data_max, data_mean, data_std = load_data('simulation_results.xlsx')
