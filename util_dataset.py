import torch.utils.data
from os.path import join, exists
import re, json, cv2, os, sys, glob
import xml.etree.ElementTree as ET
#import torchvision
import random, itertools
#from torchvision import transforms as t
#from torchvision import transforms
from PIL import Image
#import torchvision.models as models
import numpy as np

  # Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector



class MultimodalDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split: str, path: str):
        # print(path)
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: str):
        #print("js path", join(self._data_path, '{}.json'.format(i)))
        i=i.replace(" (1)", "")
        # print("iii", i)

        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        original_frames = []
        vidcap = cv2.VideoCapture(join(self._data_path, '{}.mp4'.format(i)))
        success,image = vidcap.read()
        count = 0
        while success:
            #if count % 120:
            if count % 120==0:
                #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                #original_frames.append(cv2.resize(image, (640, 360)))
                original_frames.append(torch.tensor(cv2.resize(image, (640, 360))))
            success, image = vidcap.read()
            count += 1
            if count > 7200: #reach cuda limit
                break
        # Stack it into a tensor
        if original_frames:
            video = torch.stack(original_frames, 0)
        else:
            video = torch.tensor([])

        print("video's shape: ",video.shape)
        
        
        thumbnail = cv2.imread(join(self._data_path, '{}.png'.format(i)))
        
        try:
            transcript = ET.parse(join(self._data_path, '{} (a.en).xml'.format(i))).getroot()
        except:
            transcript = ''


        return js, thumbnail, transcript, video

class EXMSMODataset(MultimodalDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, DATA_DIR):
        super().__init__(split, DATA_DIR)
        files = glob.glob(join(self._data_path, "*.json"))
        self.file_id = [os.path.split(x)[1].replace('.json', '') for x in files]

    def __getitem__(self, i):
        js, thumbnail, transcript_xml, video = super().__getitem__(self.file_id[i])

        transcript = []
        if not transcript_xml == '':
            for w in transcript_xml:
                transcript.append(w.text)
            #print("transcript_xml", transcript_xml)
            transcripts = '; '.join(transcript).replace('&#39;', '\'')
        else:
            transcripts = ''

        title = js['title']
        description= js['description']

        return  self.file_id[i], description, video, title, thumbnail, transcripts, i



class MultimodalNoTruncateDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split: str, path: str):
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: str):
        #print("js path", join(self._data_path, '{}.json'.format(i)))
        #print("i", i)

        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        original_frames = []
        vidcap = cv2.VideoCapture(join(self._data_path, '{}.mp4'.format(i)))
        vframe = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps    = vidcap.get(cv2.CAP_PROP_FPS)
        vs    = vframe/vfps

        
        thumbnail = cv2.imread(join(self._data_path, '{}.png'.format(i)))
        
        transcript = ET.parse(join(self._data_path, '{} (a.en).xml'.format(i))).getroot()


        return js, thumbnail, transcript, vframe, vfps, vs


class EXMSMONoTruncateDataset(MultimodalNoTruncateDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, DATA_DIR):
        super().__init__(split, DATA_DIR)
        files = glob.glob(join(self._data_path, "*.json"))
        self.file_id = [os.path.split(x)[1].replace('.json', '') for x in files]

    def __getitem__(self, i):
        js, thumbnail, transcript_xml, vframe, vfps, vs = super().__getitem__(self.file_id[i])

        transcript = []
        for w in transcript_xml:
            transcript.append(w.text)
        #print("transcript_xml", transcript_xml)
        transcripts = '; '.join(transcript).replace('&#39;', '\'')

        title = js['title']
        description= js['description']

        return  self.file_id[i], description, vframe, title, vfps, vs, thumbnail, transcripts



class MultimodalWithSceneDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, split: str, path: str):
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: str):
        #print("js path", join(self._data_path, '{}.json'.format(i)))
        #print("i", i)

        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())

        original_frames = []
        vidcap = cv2.VideoCapture(join(self._data_path, '{}.mp4'.format(i)))
        success,image = vidcap.read()
        count = 0
        while success:
            #if count % 120:
            if count % 360:
                #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                #original_frames.append(cv2.resize(image, (640, 360)))
                original_frames.append(torch.tensor(cv2.resize(image, (640, 360))))
            success, image = vidcap.read()
            count += 1
            if count > 100: #reach cuda limit
                break
        # Stack it into a tensor
        video = torch.stack(original_frames, 0)


        scene_list = find_scenes(join(self._data_path, '{}.mp4'.format(i)))
        thumbnail = cv2.imread(join(self._data_path, '{}.png'.format(i)))
        
        #transcript = ET.parse(join(self._data_path, '{} (a.en).xml'.format(i))).getroot()

        #return js, thumbnail, transcript, video, scene_list
        return js, thumbnail, video, scene_list

class EXMSMOWithSceneDataset(MultimodalWithSceneDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """

    def __init__(self, split, DATA_DIR):
        super().__init__(split, DATA_DIR)
        files = glob.glob(join(self._data_path, "*.json"))
        self.file_id = [os.path.split(x)[1].replace('.json', '') for x in files]

    def __getitem__(self, i):
        js, thumbnail, video, scene_list = super().__getitem__(self.file_id[i])

        #transcript = []
        #for w in transcript_xml:
        #    transcript.append(w.text)
        #print("transcript_xml", transcript_xml)
        #transcripts = '; '.join(transcript).replace('&#39;', '\'')

        title = js['title']
        description= js['description']

        #return  self.file_id[i], description, video, title, thumbnail, transcripts, scene_list
        return  self.file_id[i], description, video, title, thumbnail, scene_list

class MSMO(MultimodalDataset):
    def __init__(self, split, DATA_DIR):
        super().__init__(split, DATA_DIR)
        files = glob.glob(join(self._data_path + 'article', "*.txt"))
        self.file_id = [os.path.split(x)[1].replace('.txt', '') for x in files]

    def __getitem__(self, i):

        article_path = join(self._data_path + 'article' , '{}.txt'.format(i))
        document, extreme_summaries = get_art_abs(article_path)

        original_frames = []
        vidcap = cv2.VideoCapture(join(self._data_path, '{}.mp4'.format(i)))
        vframe = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps    = vidcap.get(cv2.CAP_PROP_FPS)
        vs    = vframe/vfps

        
        thumbnail = cv2.imread(join(self._data_path, '{}.png'.format(i)))
        
        transcript = ET.parse(join(self._data_path, '{} (a.en).xml'.format(i))).getroot()



        js, thumbnail, video, scene_list = super().__getitem__(self.file_id[i])

        #transcript = []
        #for w in transcript_xml:
        #    transcript.append(w.text)
        #print("transcript_xml", transcript_xml)
        #transcripts = '; '.join(transcript).replace('&#39;', '\'')

        title = js['title']
        description= js['description']

        #return  self.file_id[i], description, video, title, thumbnail, transcripts, scene_list
        return  self.file_id[i], description, video, title, thumbnail, scene_list


def read_story_file(text_file):
    with open(text_file, "r") as f:
        # sentences are separated by 2 newlines
        # single newlines might be image captions
        # so will be incomplete sentence
        lines = f.read().split('\n\n')
    return lines

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@summary" in line:
        return line
    if line == "":
        return line
    return line + " ."

def get_art_abs(story_file):
    """ return as list of sentences"""
    lines = read_story_file(story_file)

    # Lowercase, truncated trailing spaces, and normalize spaces
    lines = [' '.join(line.lower().strip().split()) for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem
    # in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    next_is_body = False
    for idx, line in enumerate(lines):
        #print("line", line)
        if line == "":
            #print("empty")
            continue # empty line
        elif line.startswith("@body"):
            #print("line.startswith(body)")
            next_is_body = True
            article_lines.append(line.replace("@body", '') + '.')
        elif line.startswith("@summary"):
            #print("line.startswith(summary)")
            next_is_highlight = True
            next_is_body = False
            highlights.append(line.replace("@summary", '') + '.')
        elif next_is_body:
            #print("next_is_body")
            article_lines.append(line)
        elif next_is_highlight:
            #print("next_is_highlight")
            highlights.append(line)

    return ' '.join(article_lines), ' '.join(highlights)

def find_scenes(video_path, threshold=80.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, frame_skip=360)
    # Each returned scene is a tuple of the (start, end) timecode.
    scene_list = scene_manager.get_scene_list()

    scene_frame_list = []
    for i, scene in enumerate(scene_list):
        #print(
        #    'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        #    i+1,
        #    scene[0].get_timecode(), scene[0].get_frames(),
        #    scene[1].get_timecode(), scene[1].get_frames(),))
        scene_frame_list.append(int(scene[1].get_frames()/360))
        #scene_frame_list.append(scene[1].get_frames())

    return torch.Tensor(scene_frame_list)




def _count_data(path):
    """ count number of data in the given path"""
    files = glob.glob(join(path, "*.json"))
    n_data = len(files)
    return n_data