


class sentiment_class:
    def __init__(self, seq1) -> None:
        from transformers import pipeline
        self.max_chunk = 500
        self.summarizer = pipeline("summarization")
        self.sent = pipeline(task="text-classification", model='nlptown/bert-base-multilingual-uncased-sentiment')
        self.senti = str(self.sent(seq1))

    def sentimentAnalysis(self):
        self.z = self.senti
        return self.z

    def value_to_mood(self):
        self.x = self.senti
        mood = ''
        one = self.x.find("1 star")
        two = self.x.find("2 star")
        three = self.x.find("3 star")
        four = self.x.find("4 star")
        five = self.x.find("5 star")
        if (one!=-1):
            mood = "very negative"
        if (two!=-1):
            mood = "negative"
        if (three!=-1):
            mood = "neutral"
        if (four!=-1):
            mood = "positive"
        if (five!=-1):
            mood = "very positive"
        return mood
        

class summary_class:
    

    def __init__(self, seq) -> None:
        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.modelSum = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        #self.modelSent = BartForConditionalGeneration.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.inputs = self.tokenizer([seq], max_length = 1024, return_tensors = 'pt')
        self.summary_ids = self.modelSum.generate(self.inputs['input_ids'])
        self.summary = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in self.summary_ids]
        

    def summaryExtract(self):
        self.z = self.summary
        return self.z





with open('article.txt', 'r') as file:
    x = " " 
    x = file.read().replace('\n', '')



# x = ("In May, Churchill was still generally unpopular with many Conservatives and probably most of the Labour Party. Chamberlain "
#             "remained Conservative Party leader until October when ill health forced his resignation. By that time, Churchill had won the "
#             "doubters over and his succession as party leader was a formality."
#             " "
#             "He began his premiership by forming a five-man war cabinet which included Chamberlain as Lord President of the Council, "
#             "Labour leader Clement Attlee as Lord Privy Seal (later as Deputy Prime Minister), Halifax as Foreign Secretary and Labour's "
#             "Arthur Greenwood as a minister without portfolio. In practice, these five were augmented by the service chiefs and ministers "
#             "who attended the majority of meetings. The cabinet changed in size and membership as the war progressed, one of the key "
#             "appointments being the leading trades unionist Ernest Bevin as Minister of Labour and National Service. In response to "
#             "previous criticisms that there had been no clear single minister in charge of the prosecution of the war, Churchill created "
#             "and took the additional position of Minister of Defence, making him the most powerful wartime Prime Minister in British "
#             "history. He drafted outside experts into government to fulfil vital functions, especially on the Home Front. These included "
#             "personal friends like Lord Beaverbrook and Frederick Lindemann, who became the government's scientific advisor."
#             " "
#             "At the end of May, with the British Expeditionary Force in retreat to Dunkirk and the Fall of France seemingly imminent, "
#             "Halifax proposed that the government should explore the possibility of a negotiated peace settlement using the still-neutral "
#             "Mussolini as an intermediary. There were several high-level meetings from 26 to 28 May, including two with the French "
#             "premier Paul Reynaud. Churchill's resolve was to fight on, even if France capitulated, but his position remained precarious "
#             "until Chamberlain resolved to support him. Churchill had the full support of the two Labour members but knew he could not "
#             "survive as Prime Minister if both Chamberlain and Halifax were against him. In the end, by gaining the support of his outer "
#             "cabinet, Churchill outmanoeuvred Halifax and won Chamberlain over. Churchill believed that the only option was to fight on "
#             "and his use of rhetoric hardened public opinion against a peaceful resolution and prepared the British people for a long war "
#             " Jenkins says Churchill's speeches were 'an inspiration for the nation, and a catharsis for Churchill himself'."
#             " "
#             "His first speech as Prime Minister, delivered to the Commons on 13 May was the 'blood, toil, tears and sweat' speech. It was "
#             "little more than a short statement but, Jenkins says, 'it included phrases which have reverberated down the decades'.")















###to run it##
# y = summary_class(x)
# sum = y.summaryExtract()
# print(y.summaryExtract())

# sum = str(sum)

# obj2 = sentiment(sum)
# print(obj2.sentimentAnalysis())

# text_file = open("output.txt", "w")
# n = text_file.write(str(sum))
# text_file.close()



