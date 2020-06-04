import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Model
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
from classificationMAP import getClassificationMAP as cmAP
from detectionMAP import getDetectionMAP as dmAP
import scipy.io as sio
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test(itr, dataset, args, model, logger, device):
    
    done = False
    # flow,rgb,tcam的综合分类结果
    instance_logits_stack = []  #(2380,100)
    # tcam的分类结果
    tcam_stack = []  #(2380,T,100)
    # label标签
    labels_stack = []  #(2380,100)
    
    while not done:
        if dataset.currenttestidx % 100 ==0:
            print('Testing test data point %d of %d' %(dataset.currenttestidx, len(dataset.testidx)))

        features, labels, done = dataset.load_data(is_training=False)
        seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            _, logits_f, _, logits_r, tcam, _ = model(Variable(features), device, is_training=False, seq_len=torch.from_numpy(seq_len).to(device))
            logits_f, logits_r, tcam = logits_f[0], logits_r[0], tcam[0]

        topk = int(np.ceil(len(features[0])/8))
        tmp = F.softmax(torch.mean(torch.topk(logits_f, k=topk, dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        tmp += F.softmax(torch.mean(torch.topk(logits_r, k=topk, dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        tmp += F.softmax(torch.mean(torch.topk(tcam, k=topk, dim=0)[0], dim=0), dim=0).cpu().data.numpy()
        tcam = tcam.cpu().data.numpy()        

        instance_logits_stack.append(tmp)
        tcam_stack.append(tcam)
        labels_stack.append(labels)
        
    instance_logits_stack = np.array(instance_logits_stack)   #(video_num,20)
    labels_stack = np.array(labels_stack)  # (video_num,20)

    if args.dataset_name.find('Thumos14')!= -1 and args.num_class == 101:
        test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
        bg_vid = 0
        for i in range(np.shape(labels_stack)[0]):
            if test_set[i]['background_video'] == 'YES':
                bg_vid += 1
                labels_stack[i,:] = np.zeros_like(labels_stack[i,:])

    cmap = cmAP(instance_logits_stack, labels_stack)
    print(cmap)
    # tcam_stack : (video_num,T,20),  dataset.path_to_annotations: GT 标注 , None, None 
    dmap, iou = dmAP(tcam_stack, dataset.path_to_annotations, args.activity_net, valid_id=dataset.lst_valid)
    
    print('Classification map %f' %cmap)

    for k in range(len(dmap)):
        print('Detection map @ %f = %f' %(iou[k], dmap[k]))
    
    print('Mean Detection map = %f' %(np.mean(dmap)))
    dmap += [np.mean(dmap)]    
    
    logger.log_value('Test Classification mAP', cmap, itr)
    for item in list(zip(dmap,iou)):
        logger.log_value('Test Detection mAP @ IoU = ' + str(item[1]), item[0], itr)

    utils.write_to_file(args.dataset_name + args.model_name, dmap, cmap, itr)
    
    if args.activity_net:
        return dmap.pop()
    else:
        return dmap[4]


