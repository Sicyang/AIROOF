import cv2

def get_anchor_points(image_path):
    """
    Allows the user to select anchor points on an image using mouse clicks.
    """
    img = cv2.imread(image_path)
    points = []

    def get_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", img)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", get_points)
    print("Click on the polygon vertices in the image, then press 'q' to exit.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points
