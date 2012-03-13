import cv
import time
SCALE = 4
# adjust this to tweak speed vs. accuracy for feature extraction
ROI_TARGET_SIZE = (120.0,120.0)

cascade_file_path = '/usr/local/share/opencv/haarcascades/'

cascade_files = [
    'haarcascade_frontalface_alt.xml',
    'haarcascade_mcs_eyepair_big.xml',
]

detector_names = [
    'face',
    'eyes',
]

font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, .75, .75, .12, 2)

cv.NamedWindow("Whole Image")
cv.ResizeWindow("Whole Image", *(640, 480))
cv.NamedWindow("ROI")
cv.ResizeWindow("ROI" , *(200, 200))
cv.MoveWindow("ROI" , *(640, 0))

i = 0
for name in detector_names:
    if name != 'face' and name != 'profile':
        cv.NamedWindow(name)
        cv.ResizeWindow(name , *(320, 200))
        cv.MoveWindow(name , *(0, 480+(200*i)))
        i = i+1

capture = cv.CaptureFromCAM(0)
feature_detectors = []
for file in cascade_files:
    feature_detectors.append(cv.Load("%s%s"%(cascade_file_path, file)))

def get_features(img): 
    features = {} 
    for i in range(2, len(feature_detectors)):
        if detector_names[i] not in features:
            features[detector_names[i]] = []
        features[detector_names[i]] = features[detector_names[i]] + (cv.HaarDetectObjects(img, feature_detectors[i], cv.CreateMemStorage()))
    return features
    
def get_face_roi(img): 
    frontals =  cv.HaarDetectObjects(img, feature_detectors[0], cv.CreateMemStorage())
    if len(frontals) == 0:
        profiles = cv.HaarDetectObjects(img, feature_detectors[1], cv.CreateMemStorage())
        return profiles
    return frontals

def capture_draw():
    img = cv.QueryFrame(capture)
    # scale your big ole face down to something small
    thumb = cv.CreateMat(img.height / SCALE, img.width / SCALE, cv.CV_8UC3)
    cv.Resize(img, thumb)
    faces = get_face_roi(thumb)
    for (x,y,w,h), n in faces:
        temp_offset = (x*SCALE, y*SCALE)
        cv.SetImageROI(img, ((x)*SCALE, (y)*SCALE, (w)*SCALE, (h)*SCALE))
        roi_image = cv.CreateImage(cv.GetSize(img), img.depth, img.nChannels)
        cv.Copy(img, roi_image)
        cv.ResetImageROI(img)

        cv.Rectangle(img, (x*SCALE,y*SCALE), (x*SCALE+w*SCALE, y*SCALE+h*SCALE), (255,0,0))
        cv.PutText(img, 'face', (x*SCALE, y*SCALE), font, (200,200,200))

        FEATURE_SCALE = ( float(roi_image.width) / ROI_TARGET_SIZE[0],  float(roi_image.height) / ROI_TARGET_SIZE[1])
        roi_thumb = cv.CreateImage((int(roi_image.width / FEATURE_SCALE[0]), int(roi_image.height / FEATURE_SCALE[1])), cv.IPL_DEPTH_8U, 3)
        cv.Resize(roi_image, roi_thumb)

        features = get_features(roi_thumb)
        cv.ShowImage("ROI", roi_image)
        for name in features:
            if features[name] != None:
                for (x1,y1,w1,h1), n1 in features[name]:
                    cv.SetImageROI(roi_image, (x1*FEATURE_SCALE[0],y1*FEATURE_SCALE[1],w1*FEATURE_SCALE[0],h1*FEATURE_SCALE[1]))
                    feature_image = cv.CreateImage(cv.GetSize(roi_image), roi_image.depth, roi_image.nChannels)
                    cv.Copy(roi_image, feature_image)
                    cv.ResetImageROI(feature_image)
                    cv.ShowImage(name, feature_image)
                    cv.PutText(img, name, (temp_offset[0]+x1*FEATURE_SCALE[0], temp_offset[1]+y1*FEATURE_SCALE[1]), font, (200,200,200))
                    cv.Rectangle(
                        img, 
                        (temp_offset[0]+x1*FEATURE_SCALE[0], temp_offset[1]+y1*FEATURE_SCALE[1]), 
                        (temp_offset[0]+(x1+w1)*FEATURE_SCALE[0], temp_offset[1]+(y1+h1)*FEATURE_SCALE[1]), 
                        (0,255,255)
                    )
    cv.ShowImage("Whole Image", img)

while True: capture_draw()
