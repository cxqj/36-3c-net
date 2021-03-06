import numpy as np
import time
import sys


def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def filter_segments(segment_predict, videonames, ambilist):
   ind = np.zeros(np.shape(segment_predict)[0])
   for i in range(np.shape(segment_predict)[0]):
      vn = videonames[int(segment_predict[i,0])]
      for a in ambilist:
         if a[0]==vn:
            gt = range(int(round(float(a[2])*25/16)), int(round(float(a[3])*25/16)))
            pd = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
            IoU = float(len(set(gt).intersection(set(pd))))/float(len(set(gt).union(set(pd))))
            if IoU > 0:
               ind[i] = 1
   s = [segment_predict[i,:] for i in range(np.shape(segment_predict)[0]) if ind[i]==0]
   return np.array(s)

# Inspired by Pascal VOC evaluation tool.
# 没看懂
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])  #首尾添0
    mrec = np.hstack([[0], rec, [1]])    #首尾添0和1

    # i=39021,39020,.....
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap


def getLocMAP(predictions, th, annotation_path, activity_net, valid_id):

   # gtsegments - temporal segments
   # gtlabels - labels for temporal segments
   # subset - test / validation string indicator for video
   gtsegments = np.load(annotation_path + '/segments.npy')
   gtlabels = np.load(annotation_path + '/labels.npy')
   videoname = np.load(annotation_path + '/videoname.npy'); videoname = np.array([v.decode('utf-8') for v in videoname])
   subset = np.load(annotation_path + '/subset.npy'); subset = np.array([s.decode('utf-8') for s in subset])
   classlist = np.load(annotation_path + '/classlist.npy'); classlist = np.array([c.decode('utf-8') for c in classlist])

   # 忽略的视频及其seg标注
   if not activity_net:
      ambilist = annotation_path + '/Ambiguous_test.txt'
      ambilist = list(open(ambilist,'r'))
      ambilist = [a.strip('\n').split(' ') for a in ambilist]
   else:
      gtsegments = gtsegments[valid_id]  #(7196,)
      gtlabels = gtlabels[valid_id]
      videoname = videoname[valid_id]
      subset = subset[valid_id]


   # keep training gtlabels for plotting
   # 获取训练集的gt sgements标注
   gtltr = []  # 保存每个训练视频的每个动作实例的labels
   train_str = 'validation' if not activity_net else 'training'
   for i,s in enumerate(subset):
      if subset[i]==train_str and len(gtsegments[i]):
         gtltr.append(gtlabels[i])
   gtlabelstr = gtltr
   
   # Keep only the test subset annotations
   # 获取测试集的gt segments标注
   gts, gtl, vn = [], [], []
   test_str = 'test' if not activity_net else 'validation'
   for i, s in enumerate(subset):
      if subset[i]==test_str:
         gts.append(gtsegments[i])
         gtl.append(gtlabels[i])
         vn.append(videoname[i])
   gtsegments = gts  #(2380,)
   gtlabels = gtl
   videoname = vn

   # keep ground truth and predictions for instances with temporal annotations
   # 获取测试集对应的gt segments, gt labels, video_name和预测结果
   gts, gtl, vn, pred = [], [], [], []
   for i, s in enumerate(gtsegments):
      if len(s) > 0:
         gts.append(gtsegments[i])
         gtl.append(gtlabels[i])
         vn.append(videoname[i])
         pred.append(predictions[i])
   gtsegments = gts
   gtlabels = gtl
   videoname = vn
   predictions = pred

   # which categories have temporal labels ?
   # ['Archery',....,'Zumba']
   templabelcategories = sorted(list(set([l for gtl in gtlabels for l in gtl])))

   # the number index for those categories.
   #[0,1,...,99]
   templabelidx = []
   for t in templabelcategories:
      templabelidx.append(str2ind(t,classlist))
   if len(predictions[0][0]) == 20:
      templabelidx = [i for i in range(20)]
             
   
   predictions_mod = []  # (212,T,C)只保留有效类别的预测得分结果
   c_score = []  # (212,20) , one_hot的形式，平局得分
   for i in range(len(predictions)):
      #为啥要对预测结果排序？？是为了获取有效的类别index
      pr = predictions[i]  # (T,C)
      prp = - pr; [prp[:,i].sort() for i in range(np.shape(prp)[1])]; prp=-prp
      end_id = int(np.shape(prp)[0]/8)
      if end_id == 0:
         end_id = 1
      c_s = np.mean(prp[:end_id,:],axis=0)  # (1,20)
      ind = c_s > 0 if activity_net else (c_s > np.max(c_s)/2)* (c_s > 0)  # 获取有效的类别idx
      c_score.append(c_s)
      predictions_mod.append(pr*ind)
   predictions = predictions_mod


   # For storing per-video detections (with class name, boundaries and confidence for each proposal)
   detection_results = []  # [[视频名称,类别名称，起始，结束，聚合得分]]
   for i,vn in enumerate(videoname):
      detection_results.append([])
      detection_results[i].append(vn)

   ap = []
   gtseg_c = -1
   for c in templabelidx:  # 计算每个类别的AP
      gtseg_c += 1
      segment_predict = []  #[[视频索引，起始点，结束点，聚合得分]]
      # Get list of all predictions for class c
      for i in range(len(predictions)):  # 遍历所有的视频
         tmp = predictions[i][:,c] #(T,)
         # 注意activitynet和thumos在阈值上的差异,activitynet的阈值为0
         threshold = np.max(tmp) - (np.max(tmp) - np.min(tmp))*0.5  if not activity_net else 0
         #[0,0,0,1,1,1,,0,1,1,1,0,0,0,0,1,.......0]  在开始和结束位置额外加了0
         vid_pred = np.concatenate([np.zeros(1),(tmp>threshold).astype('float32'),np.zeros(1)], axis=0)
         vid_pred_diff = [vid_pred[idt]-vid_pred[idt-1] for idt in range(1,len(vid_pred))]
         # start and end of proposals where segments are greater than the average threshold for the class
         s = [idk for idk,item in enumerate(vid_pred_diff) if item==1]  # [7,10,12,...,203]
         e = [idk for idk,item in enumerate(vid_pred_diff) if item==-1] # [8,11,13,...,205]
         for j in range(len(s)):
            # Original - Aggregate score is max value of prediction for the class in the proposal and 0.7 * mean(top-k) score of that class for the video
            aggr_score = np.max(tmp[s[j]:e[j]]) + c_score[i][c] 
            # append proposal if length is at least 2 segments (16 frames segments @ 25 fps - around 1.25 second)
            if e[j]-s[j]>=2:               
               segment_predict.append([i, s[j], e[j], aggr_score])
               detection_results[i].append([classlist[c], s[j], e[j], aggr_score])
      segment_predict = np.array(segment_predict)
      # 过滤掉和ambiist有交集的预测结果
      if not activity_net:
         segment_predict = filter_segments(segment_predict, videoname, ambilist)
   
      # Sort the list of predictions for class c based on score
      if len(segment_predict) == 0:
         return 0
      segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]  #(39021,4)

      # Create gt list
      # 获取属于该类别的所有gt segments
      segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]] for i in range(len(gtsegments)) for j in range(len(gtsegments[i])) if str2ind(gtlabels[i][j],classlist)==c]
      gtpos = len(segment_gt)  #39

      # Compare predictions and gt
      tp, fp = [], []
      for i in range(len(segment_predict)):  # 计算每一个预测结果和所有gt的交并比，如果大于某个阈值，则认为该预测结果是正样本
         flag = 0.
         best_iou = 0
         for j in range(len(segment_gt)):
            if segment_predict[i][0]==segment_gt[j][0]:  # 相同video
               # 将gt转成特征点形式
               gt = range(int(round(segment_gt[j][1]*25/16)), int(round(segment_gt[j][2]*25/16)))  
               p = range(int(segment_predict[i][1]),int(segment_predict[i][2]))
               IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
               # remove gt segment if IoU is greater than threshold (since predicted segments are sorted according to their 'actioness' scores)
               if IoU >= th:
                  flag = 1.
                  if IoU > best_iou:
                     best_iou = IoU
                     best_j = j
         if flag > 0:
            del segment_gt[best_j]
         tp.append(flag)
         fp.append(1.-flag)
      tp_c = np.cumsum(tp)  #tp:(39021)
      fp_c = np.cumsum(fp)  #fp:(39021)
      if sum(tp)==0:
         prc = 0.
      else:
         cur_prec = tp_c / (fp_c+tp_c)  #精确率  (39021,)
         cur_rec = 1. * tp_c / gtpos  #召回率    (39021,)
         prc = _ap_from_pr(cur_prec, cur_rec)
      ap.append(prc)

   return 100*np.mean(ap)
  
# prediction:(2380,T,100)
def getDetectionMAP(predictions, annotation_path, activity_net=False, valid_id=None):
   iou_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
   if activity_net:
      iou_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
   dmap_list = []

   for iou in iou_list:
      print('Testing for IoU %f' %iou)
      # predictions: (212,T,20), valid_id: 测试视频名
      dmap_list.append(getLocMAP(predictions, iou, annotation_path, activity_net, valid_id))

   return dmap_list, iou_list

