import os , subprocess ,json ,csv
from flask import Flask,render_template,request,redirect,flash,url_for
from datetime import datetime


app=Flask(__name__)
app.secret_key="ir-key"

UPLOAD_FOLDER='static/uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

#upload validater #DAY10
ALLOWED_EXTENS=['png','jpg','jpeg']
def upldvld(filename):
    is_vaild='.'in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENS
    return is_vaild

@app.route('/')
def start():
    return render_template('index.html')

@app.route('/Predict_M_Sys/static/uploads/',methods=['POST'])
def upload_files():
    if 'file'not in request.files: #checking the input name 
        flash("No file received")
        return redirect("/")
    
    file=request.files['file']  #if the file uploaded is of no nAme then return error and redirect
    if file.filename=='':
        flash("No files are selected")
        return redirect('/')
    
    if not upldvld(file.filename):
        flash("Invalid file type. Please upload a PNG or JPG file")
        return redirect('/') 

    filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename) #config the path in computer where file to be saved
    file.save(filepath)

    #calling and executing the cnn model in order to get prdeiction results and capturing the output as text
    res=subprocess.run(['python','predictor.py',filepath],capture_output=True,text=True) 
    prediction=res.stdout
    #lets log the outputs we've got
    with open('predic_logs.csv',mode='a',newline='') as logfile:
        logger=csv.writer(logfile)
        flnm=file.filename
        logger.writerow([file.filename,prediction,datetime.now().strftime("%D %H:%M")])

    # flash("Results fetched successfully")
    flash("Predictions Generated successfully ! ")
    
    return render_template('results_pg.html',prediction=prediction,flnm=flnm) #leading to the results page with the prediciton rsults

@app.route('/metrics')
def model_metrics():
    with open('model_metrics.json') as m:
        met_data=json.load(m)
        img_path=url_for('static',filename='images/confusion_matrix.jpg')
    return render_template('metrics_pg.html',met_data=met_data,confu_mat=img_path)

@app.route('/logreports')
def report():
    predicts=[]
    with open('predic_logs.csv') as lgfl:
        reder=csv.reader(lgfl)
        for eachlog in reder:
            predicts.append(eachlog)
    return render_template('logs_report_pg.html',predicts=predicts)

if __name__=="__main__":
    app.run(debug=True)
