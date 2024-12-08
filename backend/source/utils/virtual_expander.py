import numpy as np
import uuid
from tqdm import tqdm

def overlap_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if boxBArea == 0 or boxAArea == 0:
        return 0
    return max(interArea / float(boxBArea), interArea / float(boxAArea))


class Filter:
    def __init__(self, motorlist, humanlist) -> None:
        self.motorlist = motorlist
        self.humanlist = humanlist
        self.allclass = []

    def remove_overlap(self):
        list_to_remove = []
        for motor in self.motorlist:
            for motor2 in self.motorlist:
                if motor.motor_id != motor2.motor_id and overlap_ratio(motor.get_box_info(), motor2.get_box_info()) > 0.9:
                    id_to_remove = motor.motor_id if motor.conf < motor2.conf else motor2.motor_id
                    list_to_remove.append(id_to_remove)
        self.motorlist = [motor for motor in self.motorlist if motor.motor_id not in list_to_remove]

        remove_list = {}
        for human in self.humanlist:
            for human2 in self.humanlist:
                if human.human_id != human2.human_id and human.class_id == human2.class_id and overlap_ratio(human.get_box_info(), human2.get_box_info()) > 0.9:
                    if human.class_id not in remove_list:
                        remove_list[human.class_id] = []
                    id_to_remove = human.human_id if human.conf < human2.conf else human2.human_id
                    remove_list[human.class_id].append(id_to_remove)
        for key in remove_list:
            self.humanlist = [human for human in self.humanlist if human.human_id not in remove_list[key]]

    def create_virtual(self):
        self.remove_overlap()
        class_list = ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]
        for motor in self.motorlist:
            self.allclass.append(motor)
            left, top, right, bottom, class_id, conf, cls_conf = motor.get_box_info()
            for cl in class_list:
                if float(cl) != class_id:
                    self.allclass.append(Human([left, top, right - left, bottom - top, float(cl), 0.00001]))
        for human in self.humanlist:
            self.allclass.append(human)
            left, top, right, bottom, class_id, conf, cls_conf = human.get_box_info()
            for cl in class_list:
                if float(cl) != class_id:
                    self.allclass.append(Human([left, top, right - left, bottom - top, float(cl), 0.001]))
        return self.allclass


class Motor:
    def __init__(self, bbox=None, cls_conf=-1, combine_expand=0.05) -> None:
        self.left, self.top, self.width, self.height, self.class_id, self.conf = np.array(bbox).astype(float)
        self.cls_conf = cls_conf
        self.motor_id = str(uuid.uuid4().int)
        self.human_id = None
        self.humans = []
        self.heads = []
        self.right = self.width + self.left
        self.bottom = self.height + self.top
        self.combine_expand_w = combine_expand * self.width
        self.combine_expand_h = combine_expand * self.height
        self.combined_box = [self.left, self.top, self.right, self.bottom]
        self.type = "motor"

    def get_box_info(self):
        return [self.left, self.top, self.right, self.bottom, self.class_id, self.conf, self.cls_conf]


class Human(Motor):
    def init(self, bbox=None, cls_conf=-1, overlap_thres=0.3) -> None:
        super().init(bbox=bbox, cls_conf=cls_conf)
        self.human_id = str(uuid.uuid4().int)
        self.motor_id = None
        self.overlap_thres = overlap_thres
        self.wear_helmet = False
        self.x_center = (self.left + self.right) / 2
        self.heads = []
        self.type = "human"
    def attach_motor_id(self, motors: list):
        for motor in motors:
            overlap = overlap_ratio(self.get_box_info(), motor.get_box_info())
            if overlap > self.overlap_thres:
                self.motor_id = motor.motor_id
                motor.humans.append(self)
                motor.combined_box = [
                    max(0, min(self.left, motor.combined_box[0]) - self.combine_expand_w),
                    max(0, min(self.top, motor.combined_box[1]) - self.combine_expand_h),
                    min(1920, max(self.right, motor.combined_box[2]) + self.combine_expand_w),
                    min(1080, max(self.bottom, motor.combined_box[3]) + self.combine_expand_h),
                ]
                break

    def attach_head_id(self, heads: list):
        keep_heads = []
        for head in heads:
            overlap = overlap_ratio(self.get_box_info(), head.get_box_info())
            if overlap > self.overlap_thres:
                keep_heads.append(head)

        if keep_heads:
            min_dis = self.right - self.left
            nearest_head_index = -1
            for i, head in enumerate(keep_heads):
                human_head_dis = abs(self.x_center - head.x_center)
                if human_head_dis < min_dis:
                    min_dis = human_head_dis
                    nearest_head_index = i

            keep_heads[nearest_head_index].motor_id = self.motor_id
            keep_heads[nearest_head_index].human_id = self.human_id
            self.wear_helmet = keep_heads[nearest_head_index].is_helmet
            self.cls_conf = keep_heads[nearest_head_index].cls_conf
            self.heads.append(keep_heads[nearest_head_index])


class Head(Motor):
    def init(self, bbox=None, cls_conf=-1, is_helmet=False, overlap_thres=0.6) -> None:
        super().init(bbox=bbox, cls_conf=cls_conf)
        self.motor_id = None
        self.human_id = None
        self.overlap_thres = overlap_thres
        self.is_helmet = True if self.class_id == 1 else False
        self.x_center = (self.left + self.right) / 2
        self.type = "head"

    def attach_motor_id(self, motors: list, head_motor_overlap_thresh):
        avg_overlaps = []
        for motor in motors:
            head_motor_overlap = overlap_ratio(self.get_box_info(), motor.get_box_info())
            if head_motor_overlap > head_motor_overlap_thresh:
                avg_overlaps.append(0)
                continue
            overlap = overlap_ratio(self.get_box_info(), motor.combined_box)
            if overlap > self.overlap_thres:
                for human in motor.humans:
                    overlap += overlap_ratio(self.get_box_info(), human.get_box_info())
                avg_overlaps.append(overlap / len(motor.humans) if motor.humans else 0)
            else:
                avg_overlaps.append(0)
        if sum(avg_overlaps) > 0:
            max_index = avg_overlaps.index(max(avg_overlaps))
            self.motor_id = motors[max_index].motor_id
            motors[max_index].heads.append(self)

def process_objects(vid, fid, human_list, motor_list):
    filter = Filter(motor_list, human_list)
    result = []
    all_class = filter.create_virtual()
    for obj in all_class:
        left, top, right, bottom, class_id, conf, _ = obj.get_box_info()
        result.append([vid, fid, left, top, right, bottom, class_id, conf])
    return result

def process_video(dataset, vid):
    result = []
    for fid in dataset[vid].keys():
        if 'human' not in dataset[vid][fid].keys():
            dataset[vid][fid]['human'] = []
        if 'motor' not in dataset[vid][fid].keys():
            dataset[vid][fid]['motor'] = []
        result += process_objects(vid, fid, dataset[vid][fid]['human'], dataset[vid][fid]['motor'])
    return result
def Virtual_Expander(data: list):
    # print("____Start Virtual_Expander____")
    dataset = {}
    for line in data:
        vid, fid, left, top, right, bottom,_ , _, cls, conf = line
        width = right - left
        height =  bottom - top
        
        if int(float(cls)) != 1:
            
            if vid not in dataset.keys():
                dataset[vid] = {}
            if fid not in dataset[vid].keys():
                dataset[vid][fid] = {}
            if 'human' not in dataset[vid][fid].keys():
                dataset[vid][fid]['human'] = []
            dataset[vid][fid]['human'].append(Human(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
       
        else:
            if vid not in dataset.keys():
                dataset[vid] = {}
            if fid not in dataset[vid].keys():
                dataset[vid][fid] = {}
            if 'motor' not in dataset[vid][fid].keys():
                dataset[vid][fid]['motor'] = []
            dataset[vid][fid]['motor'].append(Motor(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
       
    # Create ouput
    results = []
    for vid in (dataset.keys()):
        results += process_video(dataset, vid)

    # print("____End Virtual_Expander____")
    #results[i] = [int(vid), int(fid), float(x0), float(y0),float(x1),float(y1),float(width),float(height),float(cls),float(conf)] 
    return results
