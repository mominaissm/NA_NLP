
from model import summary_class
from model import sentiment_class

from flask import Flask
app = Flask(__name__)



@app.route("/")
# def hello():
#    return "Caio!"
def main():
   with open('article.txt', 'r') as file:
    x = " " 
    x = file.read().replace('\n', '')
    y = summary_class(x)
   sum = y.summaryExtract()
   print(y.summaryExtract())

   sum = str(sum[0])


   try:
      obj2 = sentiment_class(sum)
   except:
      obj2 = sentiment_class(x)
      
   #print(obj2.sentimentAnalysis())

   text_file = open("output.txt", "w")
   n = text_file.write(str(sum))
   text_file.close()
   output = "Summary\n\n\n" + sum + "\n\n\n Sentiment:"
   output= output + obj2.value_to_mood()

   return output

if __name__ == "__main__":
   app.run()
