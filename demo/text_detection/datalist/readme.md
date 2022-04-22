## Datalist preparation for text detection and spotting

The datalist are used in text detection experiments:

- icdar2015_train_datalist.json
- icdar2015_test_datalist.json
- total_text_train_datalist.json
- total_text_test_datalist.json
- ctw1500_train_datalist_with_cares.json
- ctw1500_train_datalist_without_cares.json
- ctw1500_test_datalist_with_cares.json
- ctw1500_test_datalist_without_cares.json

All formatted datalist can be downloaded from [Link](https://one.hikvision.com/#/link/nipWaectFcwClNGrkcAT) (Access Code：o5gt)

All datalists are transformed into a unified Davar Format like:


    {
        "Images/train/img19.jpg": {
            "height": 1280, 
            "width": 960, 
            "content_ann": {
                "bboxes": [[145, 202, 322, 246, 480, 333, 449, 380, 288, 306, 114, 263], 
                           [519, 338, 695, 468, 788, 591, 740, 618, 615, 496, 473, 394]], 
                "cares": [1, 1], 
                "texts": ["Chateau", "Fujisawa"]
            }
        }, 
        "Images/train/img1102.jpg": {
            "height": 1259, 
            "width": 1280, 
            "content_ann": {
                "bboxes": [[798, 984, 861, 984, 923, 984, 923, 1050, 861, 1050, 798, 1050], 
                           [398, 984, 450, 984, 502, 984, 502, 1060, 450, 1060, 398, 1060], 
                           [334, 466, 546, 313, 792, 373, 845, 457, 764, 522, 692, 453, 560, 431, 422, 528, 420, 531], 
                           [532, 528, 641, 528, 643, 665, 554, 666], 
                           [422, 729, 580, 810, 770, 731, 838, 758, 614, 889, 373, 778]], 
                "cares": [1, 1, 0, 1, 1], 
                "texts": ["AID", "FI", "###", "6", "PATTAYA"]
            }
        },
        ...
    }