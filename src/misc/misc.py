import datetime

def datetime_now():
    """
    Return

    datetime: str
        datetime string format (YYYY-MM-DD-H-mm)
        
    """
    YEAR = datetime.date.today().year
    MONTH = datetime.date.today().month
    DATE = datetime.date.today().day
    HOUR = datetime.datetime.now().hour
    MINUTE = datetime.datetime.now().minute

    return str(YEAR)+'-'+str(MONTH)+'-'+str(DATE)+'-'+str(HOUR)+'-'+str(MINUTE)