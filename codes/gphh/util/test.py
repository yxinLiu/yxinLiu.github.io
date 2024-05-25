#encoding=utf-8

import excelUtil

if __name__ == '__main__':
    fileName = '/Users/liuyx/PycharmProjects/ga/liuCest2.xlsx'
    sheetName = 'test'
    util = excelUtil.ExcelUtil(fileName, sheetName)

    excelWb = util.createExcelSheet()
    dataStr = "124  123 123   1214  345 1231"
    data_exp = "(asdfasdfk3aadsljf,jklasdf)"
    time = "900.123.12"

    util.wirteDBToExcelByWb(dataStr, data_exp, data_exp,time)
    util.saveWbObjToExcel(excelWb)