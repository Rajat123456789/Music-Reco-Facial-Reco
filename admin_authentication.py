import pyrebase

config = {
    "apiKey": "AIzaSyCedXS9UaiCmiJMqDZhu5eNRGmgjd3LfDM",
    "authDomain": "musical-efa3d.firebaseapp.com",
    "databaseURL": "https://musical-efa3d-default-rtdb.firebaseio.com",
    "storageBucket": "musical-efa3d.appspot.com"
     }

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

database = firebase.database()

def data_upload(val):
    # email = val['email'].split('@')[0]
    # print(email)
    database.child("info").push(val)
    # print(type(email))
    # database.child("/information").child(val["email"]).set(val)

def retrieval():
    val = database.child("info").get()

    val = dict(val.val())
    l = []
    for i in val:
        l.append(val[i])
    
    return l

def user_extract(email):
    val = retrieval()
    s = ""
    for i in val:
        if i['email'] == email:
            s = i['name']
            break
    return s
    #return l

def user_extract_language(email):
    val = retrieval()
    s = ""
    for i in val:
        if i['email'] == email:
            s = i['language']
            break
    return s


def admin_login1(email, password):
    try:
        login = auth.sign_in_with_email_and_password(email, password)
        return True

    except:
        # print("Invalid email or password")
        return False


def admin_signup1(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)

        return True

    except:
        return False

def userExists(email):
    try:
        auth.get_user_by_email(email)
        return True
    except:
        return False

#   Future<bool> userAlreadyRegistered(String phoneNum) async {
#     QuerySnapshot user = await FirebaseFirestore.instance
#         .collection('users')
#         .where('phoneNum', isEqualTo: phoneNum)
#         .get();
#     if (user.docs.isEmpty) {
#       return false;
#     } else {
#       return true;
#     }
#   }


def admin_signout1():
    auth.sign_out()
    return True
def admin_signout():
    auth.sign_out()
    return True


if __name__ == "__main__":
    #data_upload(val={"name": "dereck", "age": "20"})
    retrieval()