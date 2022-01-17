#include "FileManager.h"
#include <codecvt>


FileManager::FileManager(string file_Name, Type data_Type)
{
	fstream file;
	file.open(file_Name);
	//file.imbue(locale(file.getloc(), new std::codecvt_utf8<wchar_t, 0x10FFFF, consume_header>));
	string line;

	// 컬럼명 설정
	vector<string> column_name;
	getline(file, line);
	column_name = SetColumnName(line);
	// 데이터 설정
	vector<map<int, double>> tmp_data(column_name.size());
	int row = 0;
	while (getline(file, line))
	{
		map<int, double> row_data;
		row_data = SetData(row, line);
		for (int n = 0; n < tmp_data.size(); n++)
		{
			tmp_data[n].insert({ row, row_data[n]});
		}
		row++;
	}
	// 데이터 타입 설정
	vector<Type> types;
	types = SetDataType(column_name, data_Type);

	for (int n = 0; n < column_name.size(); n++)
	{
		Data column_data;
		column_data.columnName = column_name[n];
		column_data.value = tmp_data[n];
		column_data.type = types[n];
		data.insert({ n, column_data });
	}
}

FileManager::FileManager(string file_Name, vector<Type> data_Types)
{
	fstream file;
	file.open(file_Name);
	//file.imbue(locale(file.getloc(), new std::codecvt_utf8<wchar_t, 0x10FFFF, consume_header>));
	string line;

	// 컬럼명 설정
	vector<string> column_name;
	getline(file, line);
	column_name = SetColumnName(line);
	// 데이터 설정
	vector<map<int, double>> tmp_data(column_name.size());
	int row = 0;
	while (getline(file, line))
	{
		map<int, double> row_data;
		row_data = SetData(row, line);
		for (int n = 0; n < tmp_data.size(); n++)
		{
			tmp_data[n].insert({ row, row_data[n] });
		}
		row++;
	}
	// 데이터 타입 설정
	vector<Type> types;
	types = SetDataType(column_name, data_Types);

	for (int n = 0; n < column_name.size(); n++)
	{
		Data column_data;
		column_data.columnName = column_name[n];
		column_data.value = tmp_data[n];
		column_data.type = types[n];
		data.insert({ n, column_data });
	}

}

FileManager::~FileManager()
{
	data.clear();
}

double FileManager::GetData(int row, int column)
{
	return data[column].value[row];
}

vector<double> FileManager::GetData(int row)
{
	vector<double> row_data;
	for (int n = 0; n < data.size(); n++)
	{
		row_data.push_back(data[n].value[row]);
	}

	return row_data;
}

vector<string> FileManager::SetColumnName(string first_line)
{
	vector<string> column_name;
	int col = 0;
	stringstream ss_line(first_line);
	string str_data;
	while (getline(ss_line, str_data, ','))
	{
		column_name.push_back(str_data);
		col++;
	}

	return column_name;
}

vector<FileManager::Type> FileManager::SetDataType(vector<string> column_Names, Type data_Type)
{
	vector<Type> types;
	for (string column_name : column_Names)
	{
		types.push_back(data_Type);
	}

	return types;
}

vector<FileManager::Type> FileManager::SetDataType(vector<string> column_Names, vector<Type> data_Types)
{
	vector<Type> types;
	if (column_Names.size() == data_Types.size())
	{
		for (Type data_type : data_Types)
		{
			types.push_back(data_type);
		}
	}
	else
	{
		printf("칼럼의 수와 형태의 수가 다릅니다.");
	}

	return types;
}

map<int, double> FileManager::SetData(int row, string line)
{
	map<int, double> row_data;
	int col = 0;
	stringstream ss_line(line);
	string str_data;
	while (getline(ss_line, str_data, ','))
	{
		row_data.insert({ col, stod(str_data) });
		col++;
	}

	return row_data;
}
