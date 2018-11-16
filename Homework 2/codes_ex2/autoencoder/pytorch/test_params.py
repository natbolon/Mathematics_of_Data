import os

if __name__ == '__main__':

    for epoch in ['80']:
        for hidden in ['100']:
            d = {}
            d['relu'] = '0.01'
            #d['tanh'] = '0.01'
            #d['sigmoid'] = '0.01'
            d['identity'] = '0.005'
            d['negative'] = '0.005'

            for key in d:
                act = key
                step = d[key]
                file_name = 'model' + hidden + '-' + act + '-' + step + '.ckpt'
                # os.system(
                #     'python train.py --epoch_num ' + epoch + ' --hidden_dim ' + hidden + ' --activation ' + act + \
                #    ' --step_size ' + step + ' --save_file=' + file_name)

                os.system('python reconstruct.py --hidden_dim ' + hidden + ' --activation ' + act + \
                          ' --load_file=' + file_name)
