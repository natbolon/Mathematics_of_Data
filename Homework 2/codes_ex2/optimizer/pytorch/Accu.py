import os

if __name__ == '__main__':
    outdir = 'output'
    for step in ['0.5', '1e-2', '1e-5']:
        for optimizer in ['sgd','momentumsgd', 'adam', 'adagrad', 'rmsprop']:
            print 'Step: ' + step + 'Optimizer: ' + optimizer
            file_name = optimizer + '-' + step + 'r2'
            os.system('python main.py --optimizer '+optimizer+' --learning_rate '+step+' --output=' + outdir + '/'+file_name+'.pkl')