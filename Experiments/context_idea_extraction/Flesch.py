from textstat import flesch_reading_ease
from textstat import flesch_kincaid_grade

def FleschGrade(text):
    """Return Flesch–Kincaid Grade Level (0 = early reader, 12 = high school, etc.)"""
    return flesch_kincaid_grade(text)

def Flesch(text):
    return flesch_reading_ease(text)

if __name__ == "__main__":
    print(FleschGrade("I am pleased to announce that the great Dan Scavino, in addition to remaining Deputy Chief of Staff of the Trump Administration, will head the White House Presidential Personnel Office, replacing Sergio Gor, who did a wonderful job in that position, and will now become the Ambassador to India. Dan will be responsible for the selection and appointment of almost all positions in government, a very big and important position. Congratulations Dan, you will do a fantastic job!!! President DJT"))
