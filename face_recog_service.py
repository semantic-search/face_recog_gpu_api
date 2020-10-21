import numpy as np
import globals
import face_recognition
import dlib
import face_recognition_models

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

class FaceRecog:

    def _trim_css_to_bounds(self, css, image_shape):
        """
        Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :param image_shape: numpy shape of the image array
        :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


    def _raw_face_locations(self, img, number_of_times_to_upsample=1, model="hog"):
        """
        Returns an array of bounding boxes of human faces in a image
        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                      deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of dlib 'rect' objects of found face locations
        """
        if model == "cnn":
            return cnn_face_detector(img, number_of_times_to_upsample)
        else:
            return face_detector(img, number_of_times_to_upsample)


    def _rect_to_css(self, rect):
        """
        Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
        :param rect: a dlib 'rect' object
        :return: a plain tuple representation of the rect in (top, right, bottom, left) order
        """
        return rect.top(), rect.right(), rect.bottom(), rect.left()


    def _css_to_rect(self, css):
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: a dlib `rect` object
        """
        return dlib.rectangle(css[3], css[0], css[1], css[2])


    def _raw_face_landmarks(self, face_image, face_locations=None, model="large"):
        if face_locations is None:
            face_locations = self._raw_face_locations(face_image)
        else:
            face_locations = [self._css_to_rect(face_location) for face_location in face_locations]

        pose_predictor = pose_predictor_68_point

        if model == "small":
            pose_predictor = pose_predictor_5_point

        return [pose_predictor(face_image, face_location) for face_location in face_locations]


    def face_encodings(self, face_image, known_face_locations=None, num_jitters=1, model="small"):
        """
        Given an image, return the 128-dimension face encoding for each face in the image.
        :param face_image: The image that contains one or more faces
        :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
        :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        :param model: Optional - which model to use. "large" or "small" (default) which only returns 5 points but is faster.
        :return: A list of 128-dimensional face encodings (one for each face in the image)
        """
        raw_landmarks = self._raw_face_landmarks(face_image, known_face_locations, model)
        return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
                raw_landmark_set in raw_landmarks]


    def face_distance1(self, face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param face_encodings: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)


    def face_location(self, img, number_of_times_to_upsample=0, model="hog"):
        """
        Returns an array of bounding boxes of human faces in a image
        :param img: An image (as a numpy array)
        :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
        :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
                      deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        if model == "cnn":
            return [self._trim_css_to_bounds(self._rect_to_css(face.rect), img.shape) for face in
                    self._raw_face_locations(img, number_of_times_to_upsample, "cnn")]
        else:
            return [self._trim_css_to_bounds(self._rect_to_css(face), img.shape) for face in
                    self._raw_face_locations(img, number_of_times_to_upsample, model)]


    def get_embedding(self, face_image):
        face_image = face_recognition.load_image_file(face_image)
        face_locations = self.face_location(face_image, model='cnn')
        try:
            face_encoding = self.face_encodings(face_image, face_locations, model="large")[0]
        except IndexError:
            return "No-Face"
        return face_encoding


    def face_recognition(self, face_image, encoding=False):
        embeddings = []
        for dic in globals.embeddings:
            print(dic["name"])
            embeddings.append(dic["encoding"])
        uname = None
        """if face detection occurs than try else except"""
        try:
            face_encoding = self.get_embedding(face_image)
            if face_encoding == "No-Face":
                return False
            else:
                face_distances = self.face_distance1(embeddings, face_encoding)
                result = dict()
                result["encoding"] = face_encoding
                for i, face_distance in enumerate(face_distances):
                    if face_distance < 0.6:
                        user_dic = globals.embeddings[i]
                        uname = user_dic["name"]
                        if encoding == True:
                            result["uname"] = uname
                            return result
                        else:
                            return uname
        except IndexError:
            return uname