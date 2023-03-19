string = "text(0,97,' %H_2','FontSize',9,'FontWeight','bold','color','k')" \
         "*text(-116,31,' %C_2H_6','FontSize',9,'FontWeight','bold','color','k')" \
         "*text(-75,-83,' %CH_4','FontSize',9,'FontWeight','bold','color','k')" \
         "*text(59,-83,' %C_2H_4','FontSize',9,'FontWeight','bold','color','k')" \
         "*text(95,31,' %C_2H_2','FontSize',9,'FontWeight','bold','color','k')" \
         "*text(45,43,'D1','FontSize',8,'color','k')" \
         "*text(45,-31,'D2','FontSize',8,'color','k')" \
         "*text(-35,43,'S','FontSize',8,'color','k')" \
         "*text(8,-51,'T3','FontSize',8,'color','k')" \
         "*text(-23,-51,'T2','FontSize',8,'color','k')" \
         "*text(-55,-31,'T1','FontSize',8,'color','k')" \
         "*text(-15,69,'PD','FontSize',8,'color','k')" \
         "*text(-150,-95,'PD-Partial Discharge    D1-Low Energy Discharge    D2-High Energy Discharge','FontSize',9,'FontWeight','bold','color','k')" \
         "*text(-150,-105,'T3-Thermal Faults >700C    T2-Thermal Faults of 300 to 700C','FontSize',9,'FontWeight','bold','color','k')" \
         "*text(-150,-115,'T1-Thermal Faults <300C    S-Stray Gassing of Mineral Oil','FontSize',9,'FontWeight','bold','color','k')"

string = string.split('*')


def small_text_fix(input_str):
    clsi = input_str.strip('\'').strip().strip('%')
    add_format = '%$\mathregular{' + clsi + '}$'
    #print(add_format)
    return add_format


for line in string:
    a = line.split('(')
    b = a[1].strip(')')

    c = b.split(',')

    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] == "_":
                c[i] = small_text_fix(c[i])
                # print(">>>" + c[i])
        #print(c[i])

    result = 'plt.text(' + c[0] + ', ' + c[1] + ', \'' + c[2].strip('\'') + '\', '

    for i in range(len(c)):
        if c[i] == '\'FontSize\'':
            result += 'fontsize=' + c[i + 1] + ', '
        elif c[i] == '\'FontWeight\'':
            result += 'fontweight=' + c[i + 1] + ', '
        elif c[i] == '\'color\'':
            result += 'color=' + c[i + 1] + ', '

    # print(c)

    result = result.strip(' ,')
    result += ')'
    print(result)



