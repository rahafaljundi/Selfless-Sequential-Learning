import sys
import os
import traceback
from Test_Utils import *
from Finetune_SNI_ICLR import *
from Permute_Mnist import *
num_tasks = 5
reg_lambdas = [0]
sparse_lams = [1e-3]
hsizes = [ 64]
in_layers = [['1', '3'], []]
#
reg_lambdas = [1e5, 1e6, 5e6, 1e7, 1e8]
reg_lambdas = [10, 20, 30, 40]
sparse_lams = [1e-2, 5e-3, 1e-3, 5e-3, 2e-4, 5e-4, 8e-4, 1e-4]
sparse_lams=[1e-2]
reg_lambdas = [200]
scale = 6
data_parent_path= 'Datasets'
parent_exp_dir="/esat/dragon/raljundi/SLNI_TEST/"#CHANGE TO YOUR DIRECTORY
# scale= int(sys.argv[1])
print("scale is :", str(scale))
for hsize in hsizes:
    net_name = 'permuted_' + str(hsize) + '_mnist_net'
    first_model_path = 'models/' + net_name + '.pth.tar'
    results = {}
    avg_accs = {}
    total_seq_forgetting = {}
    total_seq_acc = {}
    t = 0

    dlabel = str(t)

    dataset_path = data_parent_path + '/permuted_t' + dlabel + '_dataset.pth.tar'

    if not os.path.isfile(dataset_path):
        create_datasets()
    exp_dir =parent_exp_dir + net_name + 'SGD_MNIST' + dlabel + '_lam0ACC'

    num_epochs = 10


    for sparse_lam in sparse_lams:
        dlabel = '0'

        model_path = '/users/visics/raljundi/Code/MyOwnCode/Pytorch/my_utils/mnist_net.pth.tar'

        dataset_path = data_parent_path + '/permuted_t' + dlabel + '_dataset.pth.tar'

        exp_dir = parent_exp_dir + net_name + 'SGD_MNIST' + dlabel + 'SLNI' + '_scale' + str(
            scale) + '_lam' + str(sparse_lam)

        num_epochs = 10
        # reg_lambda=0

        fine_tune_SGD_SLNI(dataset_path=dataset_path, num_epochs=num_epochs,exp_dir=exp_dir,model_path=first_model_path,lr=0.01,in_layers=in_layers,batch_size=200,pretrained=False,weight_decay=0,init_freeze=0,lam=sparse_lam,scale=scale)
        model_path = os.path.join(exp_dir, 'best_model.pth.tar')

        # SPARSE REG

        from Finetune_EWC_SNI import *


        init_label = dlabel
        for reg_lambda in reg_lambdas:
           
            reg_sets = []
            dataset_path = data_parent_path+ '/permuted_t'+ init_label + '_dataset.pth.tar'

            try:
                for t in range(1, num_tasks):
                    dlabel = str(t)
                    reg_sets = [dataset_path]
                    model_path = os.path.join(exp_dir, 'best_model.pth.tar')

                    dataset_path = data_parent_path + '/permuted_t' + dlabel + '_dataset.pth.tar'
                    exp_dir =parent_exp_dir + net_name + 'SGD_MNIST' + dlabel + 'SLNI' + '_scale' + str(
                        scale) + '_lam' + str(sparse_lam) + '_EWC' + str(reg_lambda)
                    init_model_path = None
                    num_epochs = 10
                    data_dir = None

                    # reg_lambda=0

                    fine_tune_EWC_acuumelation_sparce(dataset_path=dataset_path,previous_task_model_path=model_path,init_model_path=init_model_path,exp_dir=exp_dir,data_dir=data_dir,reg_sets=reg_sets,reg_lambda=reg_lambda,batch_size=200,num_epochs=num_epochs,lr=1e-2,weight_decay=0,norm='L2',after_freeze=0,lam=sparse_lam,b1=False,head_shared=True,neuron_omega=True)

                # Forgetting Test:

                average_forgetting = 0
                avg_acc = 0
                seq_forgetting = []
                seq_acc = []
                for t in range(num_tasks):
                    dlabel = str(t)
                    if t == 0:
                        exp_dir = parent_exp_dir+ net_name + 'SGD_MNIST' + dlabel + 'SLNI' + '_scale' + str(
                            scale) + '_lam' + str(sparse_lam)
                    else:
                        exp_dir = parent_exp_dir + net_name + 'SGD_MNIST' + dlabel + 'SLNI' + '_scale' + str(
                            scale) + '_lam' + str(sparse_lam) + '_EWC' + str(reg_lambda)

                    previous_model_path = os.path.join(exp_dir, 'best_model.pth.tar')
                    exp_dir = parent_exp_dir+ net_name + 'SGD_MNIST' + str(
                        num_tasks - 1) + 'SLNI' + '_scale' + str(scale) + '_lam' + str(
                        sparse_lam) + '_EWC' + str(reg_lambda)
                    dataset_path = data_parent_path + '/permuted_t' + dlabel + '_dataset.pth.tar'
                    current_model_path = os.path.join(exp_dir, 'best_model.pth.tar')
                    acc2 = test_model(current_model_path, dataset_path)
                    acc1 = test_model(previous_model_path, dataset_path)
                    forgetting = acc1 - acc2
                    average_forgetting = average_forgetting + forgetting
                    avg_acc = avg_acc + acc2
                    seq_forgetting.append(forgetting)
                    seq_acc.append(acc2)
                average_forgetting = average_forgetting / num_tasks
                avg_acc = avg_acc / num_tasks
                print("avg forgetting", average_forgetting)
                print("avg acc is ",avg_acc)
                print(average_forgetting)
                results['SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                    reg_lambda)] = average_forgetting
                avg_accs['SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                    reg_lambda)] = avg_acc
                total_seq_forgetting[
                    'SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                        reg_lambda)] = seq_forgetting
                total_seq_acc[
                    'SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                        reg_lambda)] = seq_acc

            except:
                # pdb.set_trace()
                traceback.print_exc()
                total_seq_forgetting[
                    'SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                        reg_lambda)] = 0
                total_seq_acc[
                    'SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                        reg_lambda)] = 0
                results['SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                    reg_lambda)] = -1
                avg_accs['SLNI' + '_scale' + str(scale) + str(sparse_lam) + '_EWC' + str(
                    reg_lambda)] = -1
    results1 = {}
    avg_accs1 = {}
    total_seq_forgetting1 = {}
    total_seq_acc1 = {}
    try:
        results1 = torch.load('./results/5tasks_' + net_name + 'mnist_forgetting_EWC.pth.tar')
        avg_accs1 = torch.load('./results/5tasks_' + net_name + 'mnist_avg_accs_EWC.pth.tar')
        total_seq_forgetting1 = torch.load('./results/5tasks_' + net_name + 'mnist_seq_forgetting_EWC.pth.tar')
        total_seq_acc1 = torch.load('./results/5tasks_' + net_name + 'total_seq_acc_EWC.pth.tar')
    except:
        pass
    results1.update(results)
    avg_accs1.update(avg_accs)
    total_seq_forgetting1.update(total_seq_forgetting)
    total_seq_acc1.update(total_seq_acc)
    if not os.path.isdir('results'):
        os.mkdir('results')
    torch.save(results1, './results/5tasks_' + net_name + 'mnist_forgetting_EWC.pth.tar')
    torch.save(avg_accs1, './results/5tasks_' + net_name + 'mnist_avg_accs_EWC.pth.tar')
    torch.save(total_seq_forgetting1, './results/5tasks_' + net_name + 'mnist_seq_forgetting_EWC.pth.tar')
    torch.save(total_seq_acc1, './results/5tasks_' + net_name + 'total_seq_acc_EWC.pth.tar')
