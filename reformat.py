import os


def prepare_test_images_and_labels():
    test_labels_path = "dataset/test_labels"

    for image in os.listdir("test_labels"):
        with open(os.path.join("test_labels", image), "r") as orig:
            boxes = orig.readlines()
            new_boxes = []
            for bbox in boxes:
                curr = []
                for a in bbox.split():
                    curr.append(int(a))

                new_boxes.append(curr)
                print(new_boxes)

            with open(os.path.join(test_labels_path, image), "w") as f:
                for bbox in new_boxes:
                    cl, x1, y1, x2, y2 = bbox
                    w = int(abs(x2 - x1))
                    h = int(abs(y1 - y2))
                    x_center = int(x1 + w / 2)
                    y_center = int(y1 + h / 2)
                    f.write(f"0 {x_center} {y_center} {w} {h}\n")


prepare_test_images_and_labels()
