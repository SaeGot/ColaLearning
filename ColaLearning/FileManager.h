#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>


using namespace std;

class FileManager
{
public:
	enum class Type
	{
		Real,
		String
	};
	/**
	 * ��� Ÿ���� �����ϰ� ������ ����.
	 *
	 * \param file_Name : ���� �̸�
	 * \param data_Type : ������ Ÿ��
	 */
	FileManager(string file_Name, Type data_Type = Type::Real);
	/**
	 * ������ Į���� Ÿ���� �����Ͽ� ������ ����.
	 * 
	 * \param file_Name
	 * \param data_Types
	 */
	FileManager(string file_Name, vector<Type> data_Types);
	~FileManager();

private:
	vector<string> columnName;
	vector<Type> dataType;
	vector<vector<double>> data;

	/**
	 * �÷��� ����.
	 * 
	 * \param line : ù ��
	 */
	void SetColumnName(string first_line);
	/**
	 * ��� Ÿ���� �����ϰ� ����.
	 * 
	 * \param data_Type : ������ Ÿ��
	 */
	void SetDataType(Type data_Type);
	/**
	 * �� Į���� Ÿ���� ������� ����.
	 * 
	 * \param data_Types
	 */
	void SetDataType(vector<Type> data_Types);
	/**
	 * ������ ����.
	 * 
	 * \param line : ��
	 */
	void SetData(string line);
};

