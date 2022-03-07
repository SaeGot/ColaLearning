#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>


using namespace std;

class FileManager
{
public:
	enum class Type
	{
		Real,
		String,
		OneHot,
		Categorical
	};
	FileManager() {};
	/**
	 * ��� Ÿ���� �����ϰ� ������ ����.
	 *
	 * \param file_Name : ���� �̸�
	 * \param data_Type : ������ Ÿ��
	 * \param encoding_Type : ���ڵ� Ÿ��
	 */
	FileManager(string file_Name, Type data_Type = Type::Real, Type encoding_Type = Type::Categorical);
	/**
	 * ������ Į���� Ÿ���� �����Ͽ� ������ ����.
	 * 
	 * \param file_Name : ���� �̸�
	 * \param data_TypeList : ������ Ÿ��
	 * \param encoding_Type : ���ڵ� Ÿ��
	 */
	FileManager(string file_Name, vector<Type> data_Types, Type encoding_Type = Type::Categorical);
	~FileManager();
	/**
	 * �ش� ��� ������ ��������.
	 * 
	 * \param row : ��
	 * \param column : ��
	 * \return ������
	 */
	double GetData(int row, int column);
	/**
	 * �ش� �� ������ ��������.
	 * 
	 * \param row : ��
	 * \return �� ������
	 */
	vector<double> GetData(int row);
	/**
	 * �ش� Į���� ���ڵ��� �������� �ش� �� ��������.
	 * 
	 * \param row : ��
	 * \param column : ��
	 * \return ���ڵ� ������
	 */
	vector<double> GetEncodingData(int row, int column);
	/**
	 * �ش� Į������ ���ڵ��� �������� �ش� �� ��������.
	 * 
	 * \param row : ��
	 * \param columns : ��
	 * \return ���ڵ� ������
	 */
	vector<double> GetEncodingData(int row, vector<int> columns);
	/**
	 * ǥ ���·� �б�.
	 * 
	 * \param file_Name : ���� �̸�
	 * \return �࿭ ������
	 */
	vector<vector<string>> GetTable(string file_Name);
	/**
	 * �� ���� ��������.
	 * 
	 * \return �� ����
	 */
	int GetRowCount();


private:
	struct Data
	{
		string columnName;
		Type type = Type::Real;
		// <��, ��>
		map<int, string> value;
	};
	// <��, ��>
	map<int, Data> data;
	// Ÿ�� ����Ʈ
	vector<Type> typeList;
	// <Į���ε���, <��, ���ڵ�>>
	map<int, map<string, double>> encodingList;
	// <Į���ε���, <��, ���ڵ�>>
	map<int, map<double, string>> decodingList;
	// ���� Į���� ��Ī�Ǵ� ���ڵ� Į�� ����Ʈ <Į���ε���, Į����>
	map<int, vector<int>> oneHotEncodingColumnList;

	/**
	 * Į���� ����.
	 * 
	 * \param line : ù ��
	 * \return Į����
	 */
	vector<string> SetColumnName(string first_line);
	/**
	 * ��� Ÿ���� �����ϰ� ����.
	 * 
	 * \param column_Names : Į����
	 * \param data_Type : ������ Ÿ��
	 */
	vector<Type> SetDataType(vector<string> column_Names, Type data_Type);
	/**
	 * �� Į���� Ÿ���� ������� ����.
	 * 
	 * \param column_Names : Į����
	 * \param data_TypeList : ������ Ÿ��
	 */
	vector<Type> SetDataType(vector<string> column_Names, vector<Type> data_Types);
	/**
	 * ������ ����.
	 * 
	 * \param row : �� �ε���
	 * \param line : �� ������
	 */
	map<int, string> SetData(int row, string line, int column_Count);
	/**
	 * ���ڵ� ����Ʈ �߰�.
	 * 
	 * \param column : �� �ε���
	 * \param value : string �� ������ ��
	 */
	void AddEncoingList(int column, string value);
	/**
	 * String Ÿ���� ������ OneHot Encoding.
	 * 
	 * \param tmp_Data
	 * \param column_Name : Į���� ����Ʈ
	 */
	void OneHotEncoding(vector<map<int, string>>& tmp_Data, vector<string>& column_Name);
	/**
	 * String Ÿ���� ������ Categorical Encoding.
	 *
	 * \param tmp_Data
	 * \param column_Name : Į���� ����Ʈ
	 */
	void CategoricalEncoding(vector<map<int, string>>& tmp_Data, vector<string>& column_Name);
};

