import os
import sys
import webbrowser
import matplotlib.pyplot as plt
import pandas as pd

# visualization of daily attendance of a class using pie chart


def visualization(file):
    df = pd.read_csv(file)
    type_data = ["Present", "Absent"]
    a = ((df.Attendance == 'Present').sum())
    b = ((df.Attendance == 'Absent').sum())
    count_data = [a, b]
    colors = ["lightgreen", "skyblue"]
    explode = (0, 0.1)
    plt.pie(count_data, labels=type_data, explode=explode, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Attendance of the Day")
    plt.show()


# Showing Daily Attendance in Html page by highlighting absent and present


def ShowingResult(file):
    df = pd.read_csv(file)
    df = df.style.set_caption("Attendance of the Day").applymap(lambda x: 'background-color : skyblue' if x == 'Absent' else '') \
        .applymap(lambda x: 'background-color : lightgreen' if x == 'Present' else '') \
        .set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '#D2B4DE')]},
                           {'selector': 'th', 'props': [('background-color', 'yellow')]}])
    df.to_html('Attendance.html')
    filename = 'file:///' + os.getcwd() + '/' + 'Attendance.html'
    webbrowser.open_new_tab(filename)


if __name__ == '__main__':
    csv_filename = sys.argv[1]
    print(sys.argv[0])
    print(sys.argv[1])
    visualization(csv_filename)
    ShowingResult(csv_filename)


# python infovis.py Attendance_Sheet27-08-2021.csv