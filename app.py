import tensorflow as tf
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request,redirect,url_for
#from asgiref.wsgi import WsgiToAsgi
import requests

app = Flask(__name__, static_url_path='/uploads')

def tomato_pre(img_path):
#img = image.load_img(img_path, target_size=(224,224))
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

def load_rice():
    Model = load_model('Model_rice_plant_ver1.h5')
    return Model
def rice_predict(img_file,model):
    classes=['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast', 'leaf_blight', 'leaf_smut']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result


def load_tomato():
    Model = load_model('Tomato_Inception_best_v3.h5')
    return Model
def tomato_predict(img_file,model):
    classes=['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result


def load_wheat():
    Model = load_model('Wheat_VGG16.h5')
    return Model
def wheat_predict(img_file,model):
    classes=['Brown_rust', 'Healthy', 'Yellow_rust', 'septoria']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result


def load_cassava():
    Model = load_model('cassava_disease_Cnn.h5')
    return Model
def cassava_predict(img_file,model):
    classes=['Bacterial_Blight_CBB','Brown_Streak_Disease_CBSD','Green_Mottle_CGM','Healthy','Mosaic_Disease_CMD']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result

'''
def load_sugarcane():
    Model = load_model('sugarcane.h5')
    return Model
def sugarcane_predict(img_file,model):
    classes=['Sugarcane_Bacterial_Blight','Sugarcane__Healthy','Sugarcane_Red_Rot','Sugarcane_Red_Stripe','Sugarcane__rust']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result
'''


def load_potato():
    Model = load_model('potato1.h5')
    return Model
def potato_predict(img_file,model):
    classes=['Potato__Early_blight', 'Potato_Healthy', 'Potato__Late_blight']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result


def load_mango():
    Model = load_model('mango.h5')
    return Model
def mango_predict(img_file,model):
    classes=['Mango__Anthracnose','Mango_Bacterial_Canker','Mango_Cutting_Weevil','Mango_Die_Back','Mango_Gall_Midge','Mango__Healthy','Mango_Powdery_Mildew','Mango_Sooty_Mould']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result


def load_corn():
    Model = load_model('corn.h5')
    return Model
def corn_predict(img_file,model):
    classes=['Corn___Common_Rust','Corn___Gray_Leaf_Spot','Corn___Healthy','Corn___Northern_Leaf_Blight']
    img_url=img_file
    result_inception = model.predict([tomato_pre(img_url)])
#    disease=image.load_img(img_url)
    classresult=np.argmax(result_inception,axis=1)
    result=classes[classresult[0]]
    return result


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/detect/info')
def info():
    return render_template('info.html')
@app.route('/detect/tomato', methods=['GET', 'POST'])
def tomato_index():
    model=load_tomato();
    image_src ="https://i.pinimg.com/736x/56/86/a7/5686a746cc00212c68cb52a9153b103a--plant-drawing-tomato-plants.jpg"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = tomato_predict(file_path,model)
            os.remove(file_path)
            if prediction == "Tomato___Bacterial_spot":
                return redirect(url_for('Tomato_Bacterial_spot'))
            
            elif prediction == "Tomato___Early_blight":
                return redirect(url_for('Tomato_Early_blight'))
            
            elif prediction == "Tomato___Late_blight":
                return redirect(url_for('Tomato_Late_blight'))
            
            elif prediction == "Tomato___Leaf_Mold":
                return redirect(url_for('Tomato_Leaf_Mold'))
            
            elif prediction == "Tomato___Septoria_leaf_spot":
                return redirect(url_for('Tomato_Septoria_leaf_spot'))
            
            elif prediction == "Tomato___Spider_mites Two-spotted_spider_mite":
                return redirect(url_for('Tomato_Spider_mites'))
            
            elif prediction == "Tomato___Target_Spot":
                return redirect(url_for('Tomato_Target_Spot'))
            
            elif prediction == "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
                return redirect(url_for('Tomato_Tomato_Yellow_Leaf_Curl_Virus'))
            
            elif prediction == "Tomato___Tomato_mosaic_virus":
                return redirect(url_for('Tomato_Tomato_mosaic_virus'))
            
            elif prediction == "Tomato___healthy":
                return redirect(url_for('Tomato_healthy'))
            
            else:
                return render_template('crop_index.html', crop='Tomato', image_src=image_src, prediction=prediction)
#            return render_template('rice_index.html', prediction=prediction)
    return render_template('crop_index.html', crop='Tomato',image_src=image_src)


@app.route('/Tomato_Bacterial_spot')
def Tomato_Bacterial_spot():
    return render_template('Tomato_Bacterial_spot.html')

@app.route('/Tomato_Early_blight')
def Tomato_Early_blight():
    return render_template('tom_early_blight.html')

@app.route('/Tomato_Late_blight')
def Tomato_Late_blight():
    return render_template('Tomato_Late_blight.html')

@app.route('/Tomato_Leaf_Mold')
def Tomato_Leaf_Mold():
    return render_template('Tomato_Leaf_Mold.html')

@app.route('/Tomato_Septoria_leaf_spot')
def Tomato_Septoria_leaf_spot():
    return render_template('Tomato_Septoria_leaf_spot.html')

@app.route('/Tomato_Spider_mites')
def Tomato_Spider_mites():
    return render_template('Tomato_Spider_mites.html')

@app.route('/Tomato_Target_Spot')
def Tomato_Target_Spot():
    return render_template('Tomato_Target_Spot.html')

@app.route('/Tomato_Tomato_Yellow_Leaf_Curl_Virus')
def Tomato_Tomato_Yellow_Leaf_Curl_Virus():
    return render_template('Tomato_Tomato_Yellow_Leaf_Curl_Virus.html')

@app.route('/Tomato_Tomato_mosaic_virus')
def Tomato_Tomato_mosaic_virus():
    return render_template('Tomato_Tomato_mosaic_virus.html')

@app.route('/Tomato_healthy')
def Tomato_healthy():
    return render_template('Tomato_healthy.html')


@app.route('/detect/rice', methods=['GET', 'POST'])
def rice_index():
    model = load_rice()
    image_src = "https://i.pinimg.com/originals/8f/fb/be/8ffbbed8daf164e61112e13620b77fe5.jpg"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = rice_predict(file_path, model)
            os.remove(file_path)
            if prediction=="BrownSpot":
                return redirect(url_for('rice_brownspot'))
            #['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast', 'leaf_blight', 'leaf_smut']
            elif prediction=="Healthy":
                return redirect(url_for('rice_Healthy'))
            
            elif prediction=="Hispa":
                return redirect(url_for('rice_Hispa'))
            
            elif prediction=="LeafBlast":
                return redirect(url_for('rice_LeafBlast'))
            
            elif prediction=="leaf_blight":
                return redirect(url_for('rice_leaf_blight'))
            
            elif prediction=="leaf_smut":
                return redirect(url_for('rice_leaf_smut'))
                                
            else:
                return render_template('crop_index.html', crop='Rice', image_src=image_src, prediction=prediction)
    return render_template('crop_index.html', crop='Rice', image_src=image_src)

@app.route('/rice_brownspot')
def rice_brownspot():
    return render_template('Rice__Brownspot.html')

@app.route('/rice_Healthy')
def rice_Healthy():
    return render_template('rice_Healthy.html')

@app.route('/rice_Hispa')
def rice_Hispa():
    return render_template('rice_Hispa.html')

@app.route('/rice_LeafBlast')
def rice_LeafBlast():
    return render_template('rice_LeafBlast.html')

@app.route('/rice_leaf_blight')
def rice_leaf_blight():
    return render_template('rice_leaf_blight.html')

@app.route('/rice_leaf_smut')
def rice_leaf_smut():
    return render_template('rice_leaf_smut.html')


@app.route('/detect/cassava', methods=['GET', 'POST'])
def cassava_index():
    model = load_cassava()
    image_src = "https://i.pinimg.com/564x/48/2a/30/482a30960558590443784288b955b8b2.jpg"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = cassava_predict(file_path, model)
            print(prediction)
            os.remove(file_path)
            if prediction=="Bacterial_Blight_CBB":
                return redirect(url_for('cassava_Bacterial_Blight_CBB'))
            
            elif prediction=="Brown_Streak_Disease_CBSD":
                return redirect(url_for('cassava_Brown_Streak_Disease_CBSD'))
            
            elif prediction=="Green_Mottle_CGM":
                return redirect(url_for('cassava_Green_Mottle_CGM'))
            
            elif prediction=="Healthy":
                return redirect(url_for('cassava_Healthy'))
            
            elif prediction=="Mosaic_Disease_CMD":
                return redirect(url_for('cassava_Mosaic_Disease_CMD'))
                
            else:
                return render_template('crop_index.html', crop='Cassava', image_src=image_src, prediction=prediction)
    return render_template('crop_index.html', crop='Cassava', image_src=image_src)

@app.route('/cassava_Bacterial_Blight_CBB')
def cassava_Bacterial_Blight_CBB():
    return render_template('cassava_Bacterial_Blight_CBB.html')


@app.route('/cassava_Brown_Streak_Disease_CBSD')
def cassava_Brown_Streak_Disease_CBSD():
    return render_template('cassava_Brown_Streak_Disease_CBSD.html')

@app.route('/cassava_Green_Mottle_CGM')
def cassava_Green_Mottle_CGM():
    return render_template('cassava_Green_Mottle_CGM.html')

@app.route('/cassava_Healthy')
def cassava_Healthy():
    return render_template('cassava_Healthy.html')

@app.route('/cassava_Mosaic_Disease_CMD')
def cassava_Mosaic_Disease_CMD():
    return render_template('cassava_Mosaic_Disease_CMD.html')
    
    
@app.route('/detect/wheat', methods=['GET', 'POST'])
def wheat_index():
    model = load_wheat()
    image_src = "https://i.pinimg.com/474x/ba/cd/fb/bacdfb00501387732ddcb79cbb02f6a8.jpg"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = wheat_predict(file_path, model)
            os.remove(file_path)
            if prediction=="Brown_rust":
                return redirect(url_for('wheat_Brown_rust'))
            
            elif prediction=="Healthy":
                return redirect(url_for('wheat_Healthy'))
             
            elif prediction=="Yellow_rust":
                return redirect(url_for('wheat_Yellow_rust'))
             
            elif prediction=="septoria":
                return redirect(url_for('wheat_septoria'))
                
            else:
                return render_template('crop_index.html', crop='Wheat', image_src=image_src, prediction=prediction)
    return render_template('crop_index.html', crop='Wheat', image_src=image_src)

@app.route('/wheat_Brown_rust')
def wheat_Brown_rust():
    return render_template('wheat_Brown_rust.html')

@app.route('/wheat_Healthy')
def wheat_Healthy():
    return render_template('wheat_Healthy.html')

@app.route('/wheat_Yellow_rust')
def wheat_Yellow_rust():
    return render_template('wheat_Yellow_rust.html')

@app.route('/wheat_septoria')
def wheat_septoria():
    return render_template('wheat_septoria.html')

'''
@app.route('/detect/sugarcane', methods=['GET', 'POST'])
def sugarcane_index():
    model=load_sugarcane();
    image_src ="https://img.freepik.com/premium-vector/sugarcane-trees-field_642458-892.jpg?w=2000"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = sugarcane_predict(file_path,model)
            os.remove(file_path)
            if prediction=="Sugarcane_Bacterial_Blight":
                return redirect(url_for('Sugarcane_Bacterial_Blight'))
            
            elif prediction =="Sugarcane__Healthy":
                return redirect(url_for('Sugarcane__Healthy'))
                        
            elif prediction =="Sugarcane_Red_Rot":
                return redirect(url_for('Sugarcane_Red_Rot'))
            
            elif prediction =="Sugarcane_Red_Stripe":
                return redirect(url_for('Sugarcane_Red_Stripe'))
            
            elif prediction =="Sugarcane__rust":
                return redirect(url_for('Sugarcane__rust'))
            
            else:
                return render_template('crop_index.html', crop='Sugar Cane', image_src=image_src, prediction=prediction)
    return render_template('crop_index.html', crop='Sugar Cane',image_src=image_src)

'''
['Sugarcane_Bacterial_Blight',
 'Sugarcane__Healthy',
 'Sugarcane_Red_Rot',
 'Sugarcane_Red_Stripe',
 'Sugarcane__rust']
'''


@app.route('/Sugarcane_Bacterial_Blight')
def Sugarcane_Bacterial_Blight():
    return render_template('Sugarcane_Bacterial_Blight.html')


@app.route('/Sugarcane__Healthy')
def Sugarcane__Healthy():
    return render_template('Sugarcane__Healthy.html')

@app.route('/Sugarcane_Red_Rot')
def Sugarcane_Red_Rot():
    return render_template('Sugarcane_Red_Rot.html')

@app.route('/Sugarcane_Red_Stripe')
def Sugarcane_Red_Stripe():
    return render_template('Sugarcane_Red_Stripe.html')

@app.route('/Sugarcane__rust')
def Sugarcane__rust():
    return render_template('Sugarcane__rust.html')

'''
@app.route('/detect/potato', methods=['GET', 'POST'])
def potato_index():
    model=load_potato();
    image_src ="https://img.freepik.com/premium-vector/potato-plant-with-leaves-roots-vector-illustration-flat-design_8124-428.jpg?w=2000"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = potato_predict(file_path,model)
            os.remove(file_path)
            if prediction=="Potato__Early_blight":
                return redirect(url_for('Potato__Early_blight'))
            
            elif prediction =="Potato_Healthy":
                return redirect(url_for('Potato_Healthy'))
                        
            elif prediction =="Potato__Late_blight":
                return redirect(url_for('Potato__Late_blight'))
                
            else:
                return render_template('crop_index.html', crop='Potato', image_src=image_src, prediction=prediction)
    return render_template('crop_index.html', crop='Potato',image_src=image_src)
'''['Potato__Early_blight', 'Potato_Healthy', 'Potato__Late_blight']
'''


@app.route('/Potato__Early_blight')
def Potato__Early_blight():
    return render_template('Potato__Early_blight.html')


@app.route('/Potato_Healthy')
def Potato_Healthy():
    return render_template('Potato_Healthy.html')

@app.route('/Potato__Late_blight')
def Potato__Late_blight():
    return render_template('Potato__Late_blight.html')


@app.route('/detect/mango', methods=['GET', 'POST'])
def mango_index():
    model=load_mango();
    image_src ="https://i.pinimg.com/736x/67/34/2d/67342d791d409692c7143a8760a2b605.jpg"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = mango_predict(file_path,model)
            os.remove(file_path)
            if prediction=="Mango__Anthracnose":
                return redirect(url_for('Mango__Anthracnose'))
            
            elif prediction =="Mango_Bacterial_Canker":
                return redirect(url_for('Mango_Bacterial_Canker'))
                        
            elif prediction =="Mango_Cutting_Weevil":
                return redirect(url_for('Mango_Cutting_Weevil'))
            
            elif prediction =="Mango_Die_Back":
                return redirect(url_for('Mango_Die_Back'))
                        
            elif prediction =="Mango_Gall_Midge":
                return redirect(url_for('Mango_Gall_Midge'))
            
            elif prediction =="Mango__Healthy":
                return redirect(url_for('Mango__Healthy'))
            
            elif prediction =="Mango_Powdery_Mildew":
                return redirect(url_for('Mango_Powdery_Mildew'))
                
            elif prediction =="Mango_Sooty_Mould":
                return redirect(url_for('Mango_Sooty_Mould'))
                
            else:
                return render_template('crop_index.html', crop='Mango', image_src=image_src, prediction=prediction)
    return render_template('crop_index.html', crop='Mango',image_src=image_src)

'''['Mango__Anthracnose',
 'Mango_Bacterial_Canker',
 'Mango_Cutting_Weevil',
 'Mango_Die_Back',
 'Mango_Gall_Midge',
 'Mango__Healthy',
 'Mango_Powdery_Mildew',
 'Mango_Sooty_Mould']
'''


@app.route('/Mango__Anthracnose')
def Mango__Anthracnose():
    return render_template('Mango__Anthracnose.html')


@app.route('/Mango_Bacterial_Canker')
def Mango_Bacterial_Canker():
    return render_template('Mango_Bacterial_Canker.html')

@app.route('/Mango_Cutting_Weevil')
def Mango_Cutting_Weevil():
    return render_template('Mango_Cutting_Weevil.html')

@app.route('/Mango_Die_Back')
def Mango_Die_Back():
    return render_template('Mango_Die_Back.html')


@app.route('/Mango_Gall_Midge')
def Mango_Gall_Midge():
    return render_template('Mango_Gall_Midge.html')

@app.route('/Mango__Healthy')
def Mango__Healthy():
    return render_template('Mango__Healthy.html')


@app.route('/Mango_Powdery_Mildew')
def Mango_Powdery_Mildew():
    return render_template('Mango_Powdery_Mildew.html')

@app.route('/Mango_Sooty_Mould')
def Mango_Sooty_Mould():
    return render_template('Mango_Sooty_Mould.html')


@app.route('/detect/corn', methods=['GET', 'POST'])
def corn_index():
    model=load_corn();
    image_src ="https://i.pinimg.com/736x/1d/88/84/1d888456eedf9a87ca7ee3558b4c722f.jpg"
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = f"uploads/{file.filename}"
            file.save(file_path)
            prediction = corn_predict(file_path,model)
            os.remove(file_path)
            if prediction=="Corn___Common_Rust":
                return redirect(url_for('Corn___Common_Rust'))
            
            elif prediction =="Corn___Gray_Leaf_Spot":
                return redirect(url_for('Corn___Gray_Leaf_Spot'))
                        
            elif prediction =="Corn___Healthy":
                return redirect(url_for('Corn___Healthy'))
            
            elif prediction =="Corn___Northern_Leaf_Blight":
                return redirect(url_for('Corn___Northern_Leaf_Blight'))
                
            else:
                return render_template('crop_index.html', crop='Corn', image_src=image_src, prediction=prediction)
    return render_template('crop_index.html', crop='Corn',image_src=image_src)
'''['Corn___Common_Rust',
 'Corn___Gray_Leaf_Spot',
 'Corn___Healthy',
 'Corn___Northern_Leaf_Blight']
'''


@app.route('/Corn___Common_Rust')
def Corn___Common_Rust():
    return render_template('Corn___Common_Rust.html')

@app.route('/Corn___Gray_Leaf_Spot')
def Corn___Gray_Leaf_Spot():
    return render_template('Corn___Gray_Leaf_Spot.html')


@app.route('/Corn___Healthy')
def Corn___Healthy():
    return render_template('Corn___Healthy.html')

@app.route('/Corn___Northern_Leaf_Blight')
def Corn___Northern_Leaf_Blight():
    return render_template('Corn___Northern_Leaf_Blight.html')
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000,debug=False)
