from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot


bot = ChatBot(
    "Math & Time Bot",
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        "chatterbot.logic.MathematicalEvaluation",
        "chatterbot.logic.TimeLogicAdapter",
        'chatterbot.logic.LowConfidenceAdapter'
    ],
    input_adapter="chatterbot.input.VariableInputTypeAdapter",
    output_adapter="chatterbot.output.OutputAdapter"
)

# Entrenar el chatbot! Pista lean sobre ListTrainer

person_says = ["Hello", "What is 4 + 9?", "What time is it?", "Goodbye"]
bot_should_say = ["Hello", "13", "The current time is", "bye"]

for i, phrase in enumerate(person_says):
    # Get a response to the input text
    response = bot.get_response(phrase)
    print "Person:", phrase
    print "Bot:", response, "\n"
