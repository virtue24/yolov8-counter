from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.yaml")  # build a new model from YAML

if __name__ == "__main__":
    EPOCHS = 1000
    EXPERIMENT_NAME = "borek_01_01_2025"
    project_path = Path(__file__).parent / 'experiments' / EXPERIMENT_NAME
    data_yaml_path:str = input("Enter the path to the data.yaml file: ")
    model.train(
        data=data_yaml_path, 
        batch = 4, 
        epochs=EPOCHS, 
        amp=False, 
        device = 0,
        save_period	=EPOCHS//3,  
        project = project_path,   
        time = 0.25, # hours
        augment=False,  # Disable augmentation
        hsv_h=0.0,  # No hue augmentation
        hsv_s=0.0,  # No saturation augmentation
        hsv_v=0.0,  # No value (brightness) augmentation
        flipud=0.0,  # No vertical flipping
        fliplr=0.0,  # No horizontal flipping
        mosaic=0.0,  # Disable mosaic augmentation
        mixup=0.0,  # Disable mixup augmentation
        perspective=0.0,  # No perspective transformation
        scale=0.0,  # No scaling augmentation
        shear=0.0,  # No shearing augmentation
        translate=0.0,  # No translation augmentation       
    )