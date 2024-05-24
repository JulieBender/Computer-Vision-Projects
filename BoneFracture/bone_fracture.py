from ultralytics import YOLO

model = YOLO('bone-s.pt')

results = model('test_images/14f_jpg.rf.fac1f283bfc0d76d992c3a4cb9b4c9c7.jpg', show=True, save=True)