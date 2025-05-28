from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
#from pyngrok import ngrok
import os
import uuid
from classify import compute as classify_compute, load_model as load_classify_model
from count import compute as count_compute, load_model as load_count_model
from search import compute as search_compute, load_model as load_search_model
app=Flask(__name__)
app.secret_key='007jujuenidefangwen'
UPLOAD_FOLDER='static/uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
classify_model,classify_tokenizer,classify_preprocess=load_classify_model()
count_model,count_tokenizer,count_preprocess=load_count_model()
search_model, search_tokenizer, search_preprocess=load_search_model()
@app.route('/login',methods=['GET', 'POST'])
def login():
    error=None
    if request.method=='POST':
        username=request.form.get('username')
        password=request.form.get('password')
        if username=='test123' and password=='123456':
            session['logged_in']=True
            return redirect(url_for('index'))
        else:
            error='账号或密码错误'
    return render_template('login.html',error=error)
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    results=[]
    if request.method=='POST':
        task=request.form['task']
        files=request.files.getlist('images')
        image_folder=os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
        os.makedirs(image_folder, exist_ok=True)
        file_paths=[]
        for file in files:
            filepath=os.path.join(image_folder, file.filename)
            file.save(filepath)
            file_paths.append(filepath)
        if task=='classify':
            labels=request.form.get('categories', '').strip().split()
            predictions=classify_compute(classify_model, classify_tokenizer, classify_preprocess, image_folder, labels)
            results=list(zip(file_paths, predictions))
        elif task=='count':
            object_name=request.form.get('object_name', '')
            counts=count_compute(count_model, count_tokenizer, count_preprocess, image_folder, object_name)
            results=list(zip(file_paths, [f"{object_name}: {c}" for c in counts]))
        elif task=='retrieve':
            scene_name=request.form.get('scene_name', '')
            matched_files=search_compute(search_model, search_tokenizer, search_preprocess, image_folder, scene_name)
            results=[(os.path.join(image_folder, f), f"匹配: {scene_name}") for f in matched_files]
    return render_template('index.html', results=results)
@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
if __name__ == '__main__':
    port=5000
    #public_url=ngrok.connect(port,"http")
    #print(f" * 公网访问链接: {public_url}")
    app.run(host='0.0.0.0',port=port)
