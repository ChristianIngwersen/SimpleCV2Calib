from typing import Tuple
import numpy as np
import cv2
import glob
import os
import tqdm


def calibrate_camera_intrinsics(
    im_path: str, chessboard: Tuple, checkersize: int
) -> Tuple:
    """Calibrates camera based on sequence of images of a checkerboard pattern

    Args:
        im_path (str): Path to folder of either png or jpg image files
        chessboard (Tuple): inner dimensions of checkerboard
        checkersize (int): width/height of checker in mm

    Returns:
        Tuple: Tuple of intrinsic parameters and distortion parameters
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)

    # prepare object points based on checker dimensions, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard[0], 0 : chessboard[1]].T.reshape(-1, 2)
    objp *= checkersize

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(os.path.join(im_path, "*.jpg")) + glob.glob(
        os.path.join(im_path, "*.png")
    )
    for fname in tqdm.tqdm(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray, chessboard, cv2.CALIB_CB_FAST_CHECK
        )

        # If found refine and add to objectpoints
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    # Calibrate cameras based on known objectpoints and detected imagepoints
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return mtx, dist


if __name__ == "__main__":
    camera_matrix, dist_coeffs = calibrate_camera_intrinsics(
        im_path="images/", chessboard=(6, 4), checkersize=7
    )
    focallength = (camera_matrix[0, 0], camera_matrix[1, 1])
    principal_point = (camera_matrix[0, 2], camera_matrix[1, 2])

    print(f"Principal point: {principal_point}")
    print(f"Focal length: {focallength}")
