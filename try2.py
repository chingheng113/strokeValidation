import csv, os


def csv_save(selected_f):
    with open('test.csv', 'w', newline="") as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(selected_f)
