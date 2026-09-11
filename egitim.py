from ultralytics import YOLO

if __name__ == '__main__':
    # 'yolov11m.pt' yerine doğru isimlendirme olan 'yolo11l.pt' kullanıyoruz
    #model = YOLO("yolo11m.pt")
    model = YOLO(r"C:\1.Projeler\yolo-sam\runs\detect\yolov11m_geosam_model-4\weights\last.pt")


    results = model.train(
        data=r"C:\1.Projeler\yolo-sam\data.yaml",
        epochs=15,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        name="yolov11m_geosam_model",
        
        # Veri artırma (Data Augmentation)
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0
    )

    print("Eğitim tamamlandı! Ağırlıklar kaydedildi.")