string = '[RX_sample_sort{1,1};RX_sample_sort{2,1};RX_sample_sort{3,1};RX_sample_sort{4,1};RX_sample_sort{5,1};RX_sample_sort{6,1};RX_sample_sort{7,1}];'

a = string.strip('[').strip(']').split(';')
new_string = ''

for element in a:
    aa = 0
    bb = 0
    cc = 0
    if len(element) >= 5:
        for j in range(len(element)):

                if element[j] == '{':
                    aa = j
                elif element[j] == ',':
                    bb = j
                elif element[j] == '}':
                    cc = j
        first_num = element[aa+1:bb]
        new_first = str(int(first_num) - 1)
        second_num = element[bb+1:cc]
        new_second = str(int(second_num) - 1)
        new_element = element[:aa] + '[' + new_first + ',' + new_second + ']'
        new_string += new_element + ';'

print('[' + new_string.strip(';') + ']')

# b = string.replace("{", '[').replace('}', ']')
