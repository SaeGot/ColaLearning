#include "FileManager.h"
//#include <codecvt>


FileManager::FileManager(string file_Name, Type data_Type, Type encoding_Type)
{
	fstream file;
	file.open(file_Name);
	string line;

	// 컬럼명 설정
	vector<string> column_name;
	getline(file, line);
	column_name = SetColumnName(line);
	// 데이터 타입 설정
	typeList = SetDataType(column_name, data_Type);
	// 데이터 설정 (컬럼별 <row, 값>)
	vector<map<int, string>> tmp_data(column_name.size());
	int row = 0;
	while (getline(file, line))
	{
		map<int, string> row_data;
		int column_count = static_cast<int>(column_name.size());
		row_data = SetData(row, line, column_count);
		for (int n = 0; n < tmp_data.size(); n++)
		{
			tmp_data[n].insert({ row, row_data[n]});
		}
		row++;
	}
	if (encoding_Type == Type::OneHot)
	{
		OneHotEncoding(tmp_data, column_name);
	}
	else
	{
		CategoricalEncoding(tmp_data, column_name);
	}
	for (int n = 0; n < column_name.size(); n++)
	{
		Data column_data;
		column_data.columnName = column_name[n];
		column_data.type = typeList[n];
		column_data.value = tmp_data[n];
		data.insert({ n, column_data });
	}
}

FileManager::FileManager(string file_Name, vector<Type> data_Types, Type encoding_Type)
{
	fstream file;
	file.open(file_Name);
	string line;

	// 컬럼명 설정
	vector<string> column_name;
	getline(file, line);
	column_name = SetColumnName(line);
	// 데이터 타입 설정
	typeList = SetDataType(column_name, data_Types);
	// 데이터 설정 (컬럼별 <row, 값>)
	vector<map<int, string>> tmp_data(column_name.size());
	int row = 0;
	while (getline(file, line))
	{
		map<int, string> row_data;
		int column_count = static_cast<int>(column_name.size());
		row_data = SetData(row, line, column_count);
		for (int n = 0; n < tmp_data.size(); n++)
		{
			tmp_data[n].insert({ row, row_data[n] });
		}
		row++;
	}
	if (encoding_Type == Type::OneHot)
	{
		OneHotEncoding(tmp_data, column_name);
	}
	else
	{
		CategoricalEncoding(tmp_data, column_name);
	}
	for (int n = 0; n < column_name.size(); n++)
	{
		Data column_data;
		column_data.columnName = column_name[n];
		column_data.type = typeList[n];
		column_data.value = tmp_data[n];
		data.insert({ n, column_data });
	}

}

FileManager::~FileManager()
{
	data.clear();
}

double FileManager::GetData(int row, int column)
{
	return stod(data[column].value[row]);
}

vector<double> FileManager::GetData(int row)
{
	vector<double> row_data;
	for (int n = 0; n < data.size(); n++)
	{
		row_data.push_back(stod(data[n].value[row]));
	}

	return row_data;
}

vector<double> FileManager::GetEncodingData(int row, int column)
{
	vector<double> row_data;
	vector<int> encoding_columns = oneHotEncodingColumnList[column];
	for (int encoding_column : encoding_columns)
	{
		row_data.push_back(stod(data[encoding_column].value[row]));
	}

	return row_data;
}

vector<double> FileManager::GetEncodingData(int row, vector<int> columns)
{
	vector<double> row_data;
	vector<int> encoding_columns;
	for (int column : columns)
	{
		for (int encoding_column : oneHotEncodingColumnList[column])
		{
			encoding_columns.push_back(encoding_column);
		}
	}
	for (int encoding_column : encoding_columns)
	{
		row_data.push_back(stod(data[encoding_column].value[row]));
	}

	return row_data;
}

int FileManager::GetRowCount()
{
	return static_cast<int>(data.begin()->second.value.size());
}

vector<vector<string>> FileManager::GetTable(string file_Name)
{
	vector<vector<string>> table;

	fstream file;
	file.open(file_Name);
	string line;

	getline(file, line);
	vector<string> row;
	stringstream ss_line(line);
	string str_data;
	int column_count = 0;
	while (getline(ss_line, str_data, ','))
	{
		row.push_back(str_data);
		column_count++;
	}
	table.push_back(row);

	while (getline(file, line))
	{
		vector<string> row;
		stringstream ss_line(line);
		string str_data;
		for (int n = 0; n < column_count; n++)
		{
			getline(ss_line, str_data, ',');
			row.push_back(str_data);
		}
		table.push_back(row);
	}

	return table;
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
	// 기존 변수들이 인코딩형 타입이 될 수 없어, 이때는 String 타입으로 변환
	if (data_Type == Type::OneHot || data_Type == Type::Categorical)
	{
		data_Type = Type::String;
	}
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
			// 기존 변수들이 인코딩형 타입이 될 수 없어, 이때는 String 타입으로 변환
			if (data_type == Type::OneHot || data_type == Type::Categorical)
			{
				data_type = Type::String;
			}
			types.push_back(data_type);
		}
	}
	else
	{
		printf("칼럼의 수와 형태의 수가 다릅니다.");
	}

	return types;
}

map<int, string> FileManager::SetData(int row, string line, int column_Count)
{
	map<int, string> row_data;
	stringstream ss_line(line);
	string str_data;
	for (int col = 0; col < column_Count; col++)
	{
		getline(ss_line, str_data, ',');
		// 타입별 나눠서
		row_data.insert({ col, str_data });
		if (typeList[col] == Type::String)
		{
			AddEncoingList(col, str_data);
		}
	}

	return row_data;
}

void FileManager::AddEncoingList(int column, string value)
{
	if (encodingList[column].count(value) <= 0)
	{
		int oneHotNumber = static_cast<int>(encodingList[column].size());
		encodingList[column].insert({ value, static_cast<double>(oneHotNumber) });
		decodingList[column].insert({ static_cast<double>(oneHotNumber), value });
	}
}

void FileManager::OneHotEncoding(vector<map<int, string>>& tmp_Data, vector<string>& column_Name)
{
	int final_index = static_cast<int>(column_Name.size());
	for (const pair<int, map<string, double>>& encoding_list : encodingList)
	{
		// 인코딩 대상 칼럼 인덱스
		int column_index = encoding_list.first;
		// 칼럼에 해당되는 데이터
		map<int, string> column_data = tmp_Data[column_index];

		// 추가될 칼럼들
		vector<map<int, string>> new_column = vector<map<int, string>>(encoding_list.second.size());
		// 데이터 설정
		for (int row = 0; row < column_data.size(); row++)
		{
			// 인코딩 관련 모든 칼럼 0 으로 초기화 (add_col : 추가될 칼럼 인덱스)
			for (int add_col = 0; add_col < new_column.size(); add_col++)
			{
				new_column[add_col].insert({ row, "0" });
			}
			// 값에 해당하는 칼럼을 1로
			int encoding_value_index = static_cast<int>(encodingList[column_index][column_data[row]]);
			new_column[encoding_value_index][row] = "1";
		}
		// 임시 데이터에 인코딩 칼럼들 추가, 칼럼명, 칼럼타입 추가
		vector<int> encoding_column_index;
		for (int n = 0; n < new_column.size(); n++)
		{
			tmp_Data.push_back(new_column[n]);
			string add_column_name = column_Name[column_index] + "_Encoding_" + to_string(n);
			column_Name.push_back(add_column_name);
			typeList.push_back(Type::OneHot);
			// 인코딩 칼럼 인덱스 추가
			encoding_column_index.push_back(final_index);
			final_index++;
		}
		oneHotEncodingColumnList.insert({ column_index , encoding_column_index });
	}
}

void FileManager::CategoricalEncoding(vector<map<int, string>>& tmp_Data, vector<string>& column_Name)
{
	int final_index = static_cast<int>(column_Name.size());
	for (const pair<int, map<string, double>>& encoding_list : encodingList)
	{
		// 인코딩 대상 칼럼 인덱스
		int column_index = encoding_list.first;
		// 칼럼에 해당되는 데이터
		map<int, string> column_data = tmp_Data[column_index];

		// 추가될 칼럼
		map<int, string> new_column;
		// 데이터 설정
		for (int row = 0; row < column_data.size(); row++)
		{
			// 인코딩
			string data_string = tmp_Data[column_index][row];
			new_column[row] = to_string(encodingList[column_index][data_string]);
		}
		// 임시 데이터에 인코딩 칼럼들 추가, 칼럼명, 칼럼타입 추가
		vector<int> encoding_column_index;
		tmp_Data.push_back(new_column);
		string add_column_name = column_Name[column_index] + "_Encoding";
		column_Name.push_back(add_column_name);
		typeList.push_back(Type::Categorical);
		// 인코딩 칼럼 인덱스 추가
		encoding_column_index.push_back(final_index);
		final_index++;
		oneHotEncodingColumnList.insert({ column_index , encoding_column_index });
	}
}
