#include "FileManager.h"
//#include <codecvt>


FileManager::FileManager(string file_Name, Type data_Type, Type encoding_Type)
{
	fstream file;
	file.open(file_Name);
	string line;

	// �÷��� ����
	vector<string> column_name;
	getline(file, line);
	column_name = SetColumnName(line);
	// ������ Ÿ�� ����
	typeList = SetDataType(column_name, data_Type);
	// ������ ���� (�÷��� <row, ��>)
	vector<map<int, double>> tmp_data(column_name.size());
	int row = 0;
	while (getline(file, line))
	{
		map<int, double> row_data;
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

	// �÷��� ����
	vector<string> column_name;
	getline(file, line);
	column_name = SetColumnName(line);
	// ������ Ÿ�� ����
	typeList = SetDataType(column_name, data_Types);
	// ������ ���� (�÷��� <row, ��>)
	vector<map<int, double>> tmp_data(column_name.size());
	int row = 0;
	while (getline(file, line))
	{
		map<int, double> row_data;
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
	// ���� �������� ���ڵ��� Ÿ���� �� �� ����, �̶��� String Ÿ������ ��ȯ
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
			// ���� �������� ���ڵ��� Ÿ���� �� �� ����, �̶��� String Ÿ������ ��ȯ
			if (data_type == Type::OneHot || data_type == Type::Categorical)
			{
				data_type = Type::String;
			}
			types.push_back(data_type);
		}
	}
	else
	{
		printf("Į���� ���� ������ ���� �ٸ��ϴ�.");
	}

	return types;
}

map<int, double> FileManager::SetData(int row, string line, int column_Count)
{
	map<int, double> row_data;
	stringstream ss_line(line);
	string str_data;
	for (int col = 0; col < column_Count; col++)
	{
		getline(ss_line, str_data, ',');
		// Ÿ�Ժ� ������
		if (typeList[col] == Type::Real)
		{
			row_data.insert({ col, stod(str_data) });
		}
		else if (typeList[col] == Type::String)
		{
			row_data.insert({ col, StringToReal(col, str_data) });
		}
	}

	return row_data;
}

double FileManager::StringToReal(int column, string value)
{
	if (encodingList[column].count(value) <= 0)
	{
		int oneHotNumber = encodingList[column].size();
		encodingList[column].insert({ value, static_cast<double>(oneHotNumber) });
		decodingList[column].insert({ static_cast<double>(oneHotNumber), value });
	}

	return encodingList[column][value];
}

void FileManager::OneHotEncoding(vector<map<int, double>>& tmp_Data, vector<string>& column_Name)
{
	int final_index = column_Name.size();
	for (const pair<int, map<string, double>>& encoding_list : encodingList)
	{
		// ���ڵ� ��� Į�� �ε���
		int column_index = encoding_list.first;
		// Į���� �ش�Ǵ� ������
		map<int, double> column_data = tmp_Data[column_index];

		// �߰��� Į����
		vector<map<int, double>> new_column = vector<map<int, double>>(encoding_list.second.size());
		// ������ ����
		for (int row = 0; row < column_data.size(); row++)
		{
			// ���ڵ� ���� ��� Į�� 0 ���� �ʱ�ȭ (add_col : �߰��� Į�� �ε���)
			for (int add_col = 0; add_col < new_column.size(); add_col++)
			{
				new_column[add_col].insert({ row, 0 });
			}
			// ���� �ش��ϴ� Į���� 1��
			int encoding_value_index = static_cast<int>(column_data[row]);
			new_column[encoding_value_index][row] = 1;
		}
		// �ӽ� �����Ϳ� ���ڵ� Į���� �߰�, Į����, Į��Ÿ�� �߰�
		vector<int> encoding_column_index;
		for (int n = 0; n < new_column.size(); n++)
		{
			tmp_Data.push_back(new_column[n]);
			string add_column_name = column_Name[column_index] + "_Encoding_" + to_string(n);
			column_Name.push_back(add_column_name);
			typeList.push_back(Type::OneHot);
			// ���ڵ� Į�� �ε��� �߰�
			encoding_column_index.push_back(final_index);
			final_index++;
		}
		oneHotEncodingColumnList.insert({ column_index , encoding_column_index });
	}
}

void FileManager::CategoricalEncoding(vector<map<int, double>>& tmp_Data, vector<string>& column_Name)
{
	int final_index = column_Name.size();
	for (const pair<int, map<string, double>>& encoding_list : encodingList)
	{
		// ���ڵ� ��� Į�� �ε���
		int column_index = encoding_list.first;
		// Į���� �ش�Ǵ� ������
		map<int, double> column_data = tmp_Data[column_index];

		// �߰��� Į��
		map<int, double> new_column;
		// ������ ����
		for (int row = 0; row < column_data.size(); row++)
		{
			// ���ڵ� ���� ��� Į�� 0 ���� �ʱ�ȭ (add_col : �߰��� Į�� �ε���)
			new_column.insert({ row, 0 });
			// ���� �ش��ϴ� Į���� 1��
			int encoding_value_index = static_cast<int>(column_data[row]);
			new_column[row] = encoding_value_index;
		}
		// �ӽ� �����Ϳ� ���ڵ� Į���� �߰�, Į����, Į��Ÿ�� �߰�
		vector<int> encoding_column_index;
		tmp_Data.push_back(new_column);
		string add_column_name = column_Name[column_index] + "_Encoding";
		column_Name.push_back(add_column_name);
		typeList.push_back(Type::Categorical);
		// ���ڵ� Į�� �ε��� �߰�
		encoding_column_index.push_back(final_index);
		final_index++;
		oneHotEncodingColumnList.insert({ column_index , encoding_column_index });
	}
}
