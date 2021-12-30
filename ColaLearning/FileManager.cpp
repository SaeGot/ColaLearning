#include "FileManager.h"


FileManager::FileManager(string file_Name, Type data_Type)
{
	fstream file;
	file.open(file_Name);
	string line;

	getline(file, line);
	// 컬럼명 설정
	SetColumnName(line);
	// 데이터 타입 설정
	SetDataType(data_Type);
	// 데이터 설정
	while (getline(file, line))
	{
		SetData(line);
	}
}

FileManager::FileManager(string file_Name, vector<Type> data_Types)
{
	// ToDo
}

FileManager::~FileManager()
{
	columnName.clear();
	data.clear();
}

void FileManager::SetColumnName(string first_line)
{
	int col = 0;
	stringstream ss_line(first_line);
	string str_data;
	while (getline(ss_line, str_data, '\t'))
	{
		columnName.push_back(str_data);
		col++;
	}
}

void FileManager::SetDataType(Type data_Type)
{
	for (int n = 0; n < columnName.size(); n++)
	{
		dataType.push_back(data_Type);
	}
}

void FileManager::SetDataType(vector<Type> data_Types)
{
	// ToDo
}

void FileManager::SetData(string line)
{
	int col = 0;
	vector<double> line_data;
	stringstream ss_line(line);
	string str_data;
	while (getline(ss_line, str_data, '\t'))
	{
		line_data.push_back(stod(str_data));
		col++;
	}
	data.push_back(line_data);
}
