from flask import Flask
from flask import render_template 
from flask import Flask, redirect, url_for, request, redirect
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import graphviz
from collections import Counter
import csv





app = Flask(__name__)



@app.route('/startseite')
def startseite():
    name = 'Sebastian'
    return render_template('startseite.html', username= name, title='Startseite')

@app.route('/')
@app.route('/seite', methods=['POST','GET'])
def seite():
    
    return render_template('seite.html', title='Startseite')


@app.route('/fragebogen')
def fragebogen():
    name = 'Select the following features and get the most likely corneal disease!'
    return render_template('fragebogen.html', title='Welcome', subtitle=name)

@app.route("/adminmode", methods=['POST','GET'])
def admin():
       
    return render_template('admin.html')     


#adminmode loading page 
@app.route("/entrycreated", methods=['POST','GET'])
def entry():
    if request.method == 'GET':
        
        meinevariable = request.args.get('frage1')
        meinevariable2 = request.args.get('frage2')
        meinevariable3 = request.args.get('frage3')
        meinevariable4 = request.args.get('frage4')
        meinevariable5 = request.args.get('frage5')
        meinevariable6 = request.args.get('frage6')
        meinevariable7 = request.args.get('frage7')
        meinevariable8 = request.args.get('frage8')
        meinevariable9 = request.args.get('frage9')
        meinevariable10 = request.args.get('frage10')
        meinevariable11 = request.args.get('frage11')
        meinevariable12 = request.args.get('frage12')
        meinevariable13 = request.args.get('frage13')
        meinevariable14 = request.args.get('frage14')
        meinevariable15 = request.args.get('frage15')
        meinevariable16 = request.args.get('frage16')
        meinevariable17 = request.args.get('frage17')

        # opening the csv file in 'w' mode
        file = open('user_input_data.csv', 'w', newline ='')
        
        with file:
            # identifying header 
            header = ['Name','IC3D-Code','omim','orphacode','decade of diagnosis','primarily affected layer','recurrent erosions','corneal thinning','non progressive','inheritance','may be unilateral','microcysts','epithelial thickening','stroma: rings / stars','stroma: central snowflakes / lines','stroma: cloudy appearance','stroma: arcus','stroma: honeycomb','stroma: confluent geographic','stroma: pre decemetal haze']
            writer = csv.DictWriter(file, fieldnames = header)
            
            # writing data row-wise into the csv file
            writer.writeheader()
            writer.writerow({'Name' : meinevariable17,
                            'IC3D-Code': 'nan',
                            'omim': 'nan',
                            'orphacode': 'nan',
                            'decade of diagnosis': meinevariable,
                            'primarily affected layer': meinevariable3,
                            'recurrent erosions': meinevariable2,
                            'corneal thinning': meinevariable4, 
                            'non progressive': meinevariable5, 
                            'inheritance': meinevariable6, 
                            'may be unilateral': meinevariable7, 
                            'microcysts': meinevariable8,
                            'epithelial thickening': meinevariable9, 
                            'stroma: rings / stars': meinevariable10, 
                            'stroma: central snowflakes / lines': meinevariable11, 
                            'stroma: cloudy appearance': meinevariable12, 
                            'stroma: arcus': meinevariable13, 
                            'stroma: honeycomb': meinevariable14, 
                            'stroma: confluent geographic': meinevariable15, 
                            'stroma: pre decemetal haze': meinevariable16 })

            # list of column names
            field_names = header
 
            # Dictionary that we want to add as a new row
            dict = {'Name' : meinevariable17,
                            'IC3D-Code': 'nan',
                            'omim': 'nan',
                            'orphacode': 'nan',
                            'decade of diagnosis': meinevariable,
                            'primarily affected layer': meinevariable3,
                            'recurrent erosions': meinevariable2,
                            'corneal thinning': meinevariable4, 
                            'non progressive': meinevariable5, 
                            'inheritance': meinevariable6, 
                            'may be unilateral': meinevariable7, 
                            'microcysts': meinevariable8,
                            'epithelial thickening': meinevariable9, 
                            'stroma: rings / stars': meinevariable10, 
                            'stroma: central snowflakes / lines': meinevariable11, 
                            'stroma: cloudy appearance': meinevariable12, 
                            'stroma: arcus': meinevariable13, 
                            'stroma: honeycomb': meinevariable14, 
                            'stroma: confluent geographic': meinevariable15, 
                            'stroma: pre decemetal haze': meinevariable16 }
            
            # Open CSV file in append mode
            # Create a file object for this file
            with open('nutzer_input.csv', 'a') as f_object:
            
                # Pass the file object and a list
                # of column names to DictWriter()
                # You will get a object of DictWriter
                dictwriter_object = csv.DictWriter(f_object, fieldnames=field_names)
            
                # Pass the dictionary as an argument to the Writerow()
                dictwriter_object.writeheader()
                dictwriter_object.writerow(dict)
            
                # Close the file object
                f_object.close()
            
        

    return render_template('entrycreated.html', eins = meinevariable, zwei= meinevariable2, drei=meinevariable3, vier=meinevariable4, fuenf=meinevariable5, sechs=meinevariable6, sieben=meinevariable7, acht=meinevariable8, neun=meinevariable9, zehn=meinevariable10, elf=meinevariable11, zwoelf=meinevariable12, dreizehn=meinevariable13, vierzehn=meinevariable14, fuenfzehn=meinevariable15, sechszehn=meinevariable16, siebzehn=meinevariable17)      



@app.route("/submitted", methods=['POST', 'GET'])
def submitted():
   if request.method == 'GET':
        ergebnisliste = []
        myvariable = request.args.get('question1')
        myvariable2 = request.args.get('question2')
        myvariable3 = request.args.get('question3')
        myvariable4 = request.args.get('question4')
        myvariable5 = request.args.get('question5')
        myvariable6 = request.args.get('question6')
        myvariable7 = request.args.get('question7')
        myvariable8 = request.args.get('question8')
        myvariable9 = request.args.get('question9')
        myvariable10 = request.args.get('question10')
        myvariable11 = request.args.get('question11')
        myvariable12 = request.args.get('question12')
        myvariable13 = request.args.get('question13')
        myvariable14 = request.args.get('question14')
        myvariable15 = request.args.get('question15')
        myvariable16 = request.args.get('question16')
        ergebnisliste = [myvariable, myvariable2 ,myvariable3 ,myvariable4, myvariable5, myvariable6, myvariable7,myvariable8,myvariable9,myvariable10,myvariable11, myvariable12,myvariable13,myvariable14,myvariable15, myvariable16]
        #frage_1 decade of diagnosis
        
        
        def transform_ergebnisse(liste):
            prediction_list = []

            if liste[0] == "0":
                prediction_list.append([1,0,0,0,0])
            elif liste[0] == '1':
                prediction_list.append([0,1,0,0,0])    
            elif liste[0] == '2':
                prediction_list.append([0,0,1,0,0])    
            elif liste[0] == '3':
                prediction_list.append([0,0,0,1,0])
            elif liste[0] == 'unknown':
                prediction_list.append([0,0,0,0,1])        

            #frage_2 = input('keine Wiederkehrenden erosionen, wiederkehrende erosionen, unbekannt')
            if liste[1] == "No":
                prediction_list.append([1,0,0])
            elif liste[1] == 'Yes':
                prediction_list.append([0,1,0])    
            elif liste[1] == 'unknown':
                prediction_list.append([0,0,1]) 
            
            #frage_3 = input('endo, epi, stro, stro, endo, unbekannt')
            if liste[2] == "Endothelium":
                prediction_list.append([1,0,0,0,0])
            elif liste[2] == 'Epithelium':
                prediction_list.append([0,1,0,0,0])    
            elif liste[2] == 'Stroma':
                prediction_list.append([0,0,1,0,0])    
            elif liste[2] == 'Stroma/Endo':
                prediction_list.append([0,0,0,1,0])  
            elif liste[2] == 'unknown':
                prediction_list.append([0,0,0,0,1]) 
                
            #frage_4 = input('keine Korneaausdünnung, unbekannt')
            
            if liste[3] == "Yes":
                prediction_list.append([1,0])
            elif liste[3] == "No":
                prediction_list.append([0,1])     
            elif liste[3] == 'unknown':
                prediction_list.append([0,0])    
            
            #frage_5 = input('nicht progressiv, unbekannt ')
            if liste[4] == "No":
                prediction_list.append([1,0])
            elif liste[4] == "Yes":
                prediction_list.append([0,1])    
            elif liste[4] == 'unknown':
                prediction_list.append([0,0])       
                    
            #frage_6 = input('Vererbung:  AD,  AR,  NO,  X,  XD, XR')
            if liste[5] == "AD":
                prediction_list.append([1,0,0,0,0,0])
            elif liste[5] == 'AR':
                prediction_list.append([0,1,0,0,0,0])    
            elif liste[5] == 'NO':
                prediction_list.append([0,0,1,0,0,0])    
            elif liste[5] == 'X':
                prediction_list.append([0,0,0,1,0,0])  
            elif liste[5] == 'XD':
                prediction_list.append([0,0,0,0,1,0])
            elif liste[5] == 'XR':
                prediction_list.append([0,0,0,0,0,1])
            elif liste[5] == 'unknown':
                prediction_list.append([0,0,0,0,0,0])         
            
            #frage_7 = input('unilateral, unbekannt')
            if liste[6] == "Yes":
                prediction_list.append([1,0])
            elif liste[6] == 'No':
                prediction_list.append([0,1])
            elif liste[6] == 'unknown':
                prediction_list.append([0,0])      
            
            #frage_8 = input('mikrozysten, unbekannt')
            if liste[7] == "Yes":
                prediction_list.append([1,0])
            elif liste[7] == 'No':
                prediction_list.append([0,1]) 
            elif liste[7] == 'unknown':
                prediction_list.append([0,0]) 

            #frage_9 = input('Epithelverdickung, unbekannt')
            if liste[8] == "Yes":
                prediction_list.append([1,0])
            elif liste[8] == 'No':
                prediction_list.append([0,1])
            elif liste[8] == 'unknown':
                prediction_list.append([0,0])          

            #frage_95 = input('stroma: rings / stars   , unbekannt ')
            if liste[9] == "Yes":
                prediction_list.append([1,0])
            elif liste[9] == 'No':
                prediction_list.append([0,1])
            elif liste[9] == 'unknown':
                prediction_list.append([0,0])     


            #frage_10 = input('stroma: central snowflakes,  unbekannt')
            if liste[10] == "Yes":
                prediction_list.append([1,0])
            elif liste[10] == 'No':
                prediction_list.append([0,1])
            elif liste[10] == 'unknown':
                prediction_list.append([0,0])      

            #frage_11 = input('stroma: cloudy appearance, unbekannt')
            if liste[11] == "Yes":
                prediction_list.append([1,0])
            elif liste[11] == 'No':
                prediction_list.append([0,1])
            elif liste[11] == 'unknown':
                prediction_list.append([0,0])          


            #frage_12 = input('stroma: arcus_stroma, unbekannt ')
            if liste[12] == "Yes":
                prediction_list.append([1,0])
            elif liste[12] == 'No':
                prediction_list.append([0,1])
            elif liste[12] == 'unknown':
                prediction_list.append([0,0])     

            #frage_13 = input('stroma: honeycomb, unbekannt')
            if liste[13] == "Yes":
                prediction_list.append([1,0])
            elif liste[13] == 'No':
                prediction_list.append([0,1])
            elif liste[13] == 'unknown':
                prediction_list.append([0,0])      

            #frage_14 = input('stroma: confluent geographic, unbekannt')
            if liste[14] == "Yes":
                prediction_list.append([1,0])
            elif liste[14] == 'No':
                prediction_list.append([0,1]) 
            elif liste[14] == 'unknown':
                prediction_list.append([0,0])     

            #frage_15 = input('stroma: pre decemetal haze, unbekannnt ')
            if liste[15] == "Yes":
                prediction_list.append([1,0])
            elif liste[15] == 'No':
                prediction_list.append([0,1]) 
            elif liste[15] == 'unknown':
                prediction_list.append([0,0])         
            
            return prediction_list    


        # erzeuge liste mit einzelnen listen als elemente
        liste = transform_ergebnisse(ergebnisliste)   

        #erzeuge ergebnisliste (nur mit einzelwerten)
        neue_liste = []

        for i in range(16):
            neue_liste = neue_liste + liste[i]

        #neue_liste = sum (liste, [])    

        # BEGINN DATENBLATT UND MODELLIERUNG

        #Lade csv 
        daten = pd.read_csv('corneal_dystrophies _data Kopie.csv')


        daten = daten.fillna('unknown')
        daten = daten.replace('y', 'yes')
        daten = daten.replace('n', 'no')
        #daten.head(5)


        features = list(daten.head(0))
        features = features[4::]


        #Modifiziere Eingabe

        for i in features: 
            daten[i] = daten[i].replace('yes', i)
            daten[i] = daten[i].replace('no', f'not_{i}')
            daten[i] = daten[i].replace('unknown', f'unknown_{i}')

        
        # One-Hot encoding aller features

        daten_encoded = pd.get_dummies(data=daten, columns=['decade of diagnosis','recurrent erosions', 'primarily affected layer','corneal thinning', 'non progressive','inheritance', 'may be unilateral', 'microcysts', 'epithelial thickening', 'stroma: rings / stars', 'stroma: central snowflakes / lines', 'stroma: cloudy appearance', 'stroma: arcus', 'stroma: honeycomb', 'stroma: confluent geographic', 'stroma: pre decemetal haze']) 
        daten_encoded
        
        #daten_encoded.to_csv('daten_encoded.csv')


        # ergebnisvektor
        y = daten["Name"]
        target_array = y.values
        target_array


        namen = list(daten_encoded.head(0))
       
        namen = namen[4::]



        X = daten_encoded[namen].values
        y = target_array

        #Decision Tree
        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

        model = DecisionTreeClassifier(random_state = 1)
        model.fit(X, y)


        #Random Forest
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)

        #k nearest Neighbours

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)

        #Logistic Rergression

        log = LogisticRegression()
        log.fit(X,y)

        #categorial bayes
        clf = CategoricalNB()
        clf.fit(X, y)

        #multi layer perceptron
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        mlp.fit(X,y)
        
        #support vector machines
        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(X,y)

        
        ergebnisse = []

        tree = str(model.predict([neue_liste])).strip('[]')

        forest = str(clf.predict([neue_liste])).strip('[]')

        neighbors = str(neigh.predict([neue_liste])).strip('[]')

        regression = str(log.predict([neue_liste])).strip('[]')

        bayesian = str(clf.predict([neue_liste])).strip('[]')

        perception = str(mlp.predict([neue_liste])).strip('[]')
       
        support = str(clf_svm.predict([neue_liste])).strip('[]')

        ergebnisse = [tree, forest, regression, bayesian, perception, support]
        
        ergebnisse = list(ergebnisse)
        c = Counter(ergebnisse)
        haeufigkeit = c.most_common()
        
        first_place = haeufigkeit[0]
        second_place = haeufigkeit[1]
        
        first_place = first_place[0]
        second_place = second_place[0]

        first_place = first_place.strip('[]')
        second_place = second_place.strip('[]')

        



        # Predict Decision tree 


        #print('Das Decision tree - Model sagt: ' + model.predict([neue_liste]))

        #predict random forest

        #print('Das Random Forest - Model sagt: ' + clf.predict([neue_liste]))

        #predict k nearest

       # print('Das K-nearest Neighbors Model sagt: ' + neigh.predict([neue_liste]))

        #prediict log regression

        #print('Das log regression Model sagt: ' + log.predict([neue_liste]))

        #variables = [model.predict([neue_liste]), clf.predict([neue_liste]), neigh.predict([neue_liste]), log.predict([neue_liste])]
        


        #return renderf'Decision Tree: {model.predict([neue_liste])}, Random Forest: {clf.predict([neue_liste])}, k-nearest neigh: {neigh.predict([neue_liste])}, logistic regression: {log.predict([neue_liste])}'      #render_template ('startseite.html', username = myvariable, title = 'submitted')
        return render_template('ergebnis.html', decision_tree=tree , random_forest =forest, k_neigh=neighbors, log_reg= regression, bayes= bayesian, perceptron= perception, supportvector= support, erster_platz = first_place, zweiter_platz= second_place )

@app.route('/tabelle')
def tabelle(): 
    filename = 'corneal_dystrophies _data Kopie.csv' 
 
	# to read the csv file using the pandas library 
    data = pd.read_csv(filename) 
    werte = list(data.head(0))
 
    myData = data.values
    return render_template('tabelle.html', myData=myData, werte =werte)

@app.route('/nutzerinput')
def nutzerinput(): 
    filename = 'nutzer_input.csv' 
 
	# to read the csv file using the pandas library 
    data = pd.read_csv(filename) 
    werte = list(data.head(0))
 
    myData = data.values
    return render_template('tabelle.html', myData=myData, werte =werte)

@app.route('/fig/<cropzonekey>')
def fig(cropzonekey):
    fig = draw_polygons(cropzonekey)
    img = StringIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81, debug=True)


