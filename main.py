from fastapi import FastAPI, File, UploadFile, Form
from db_models.mongo_setup import global_init
from db_models.models.user_model import UserModel
from db_models.models.face_model import FaceModel
import os
import globals
import pickle
from face_recog_service import FaceRecog
import base64


face_recog_obj = FaceRecog()


def _save(file):
    file_name = file.filename
    with open(file_name, 'wb') as f:
        f.write(file.file.read())
    return file_name


app = FastAPI()

origins = [
    globals.CORS_ORIGIN
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_init()
for user in UserModel.objects:
    globals.add_to_embeddings(username=user.user_name, encoding=pickle.loads(user.encoding))



@app.post("/register/")
def register(file: UploadFile = File(...), user_name: str = Form(...)):
    try:
        UserModel.objects.get(user_name=user_name)
        print("############USER ALREADY EXISTS ################")
        return False
    except UserModel.DoesNotExist:
        """If user_name not in db than error will be handled here """
        user_model_obj = UserModel()
        file_name = _save(file)
        face_encoding = face_recog_obj.get_embedding(file_name)
        if face_encoding == "No-Face":
            print("############NO FACE DETECTED################")
            return False
        else:
            binary_encoding = pickle.dumps(face_encoding)
            user_model_obj.user_name = user_name
            user_model_obj.encoding = binary_encoding
            """saving data in db through model ob of user"""
            with open(file_name, 'rb') as fd:
                user_model_obj.image.put(fd)
            os.remove(file_name)
            user_model_obj.save()
            return True


@app.post("/recognize")
def recog(file: UploadFile = File(...)):
    file_name = _save(file)
    uname = face_recog_obj.face_recognition(file_name)
    if uname is None:
        """if unknown person detected than return none"""
        os.remove(file_name)
        return False
    elif uname is False:
        """if unknown person detected than return none"""
        os.remove(file_name)
        return False

    else:
        print("in else")
        print("*************")
        print(uname)
        print("#############")
        return uname


def fetch_images(uname):
    print(uname)
    b64_data = list()
    doc_ids = list()
    face_objects = FaceModel.objects(person=uname)
    for obj in face_objects:
        image = obj.file.read()
        doc_id = str(obj.document.id)
        b64_file = base64.b64encode(image)
        if b64_file is None or not b64_file:
            print("********************IN EMPTY**************************************")
            obj.file.seek(0)
            image = obj.file.read()
            b64_file = base64.b64encode(image)
        b64_data.append(b64_file)
        doc_ids.append(doc_id)
    data = {
        "document": doc_ids,
        "files": b64_data
    }
    return data


@app.post("/face_search")
def face_search(file: UploadFile = File(...)):
    file_name = _save(file)
    uname = face_recog_obj.face_recognition(file_name)
    if uname is not None:
        data = fetch_images(uname)
        return data
    else:
        return False


@app.post("/person_search")
def face_search(user_name: str = Form(...)):
    data = fetch_images(user_name)
    b64_list = data["files"]
    if len(b64_list) == 0:
        return False
    else:
        return data
