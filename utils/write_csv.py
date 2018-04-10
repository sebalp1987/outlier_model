import csv

class WriteCsv:

    def write_csv(input: list, output_file_path, header = None):

        with open(output_file_path,'w', newline='') as file:
            wr = csv.writer(file, delimiter = ';', quoting = csv.QUOTE_ALL)
            if header:
                wr.writerow(header)

            for i in input:
                wr.writerow(i)



