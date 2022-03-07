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
	 * 모든 타입을 동일하게 데이터 생성.
	 *
	 * \param file_Name : 파일 이름
	 * \param data_Type : 데이터 타입
	 * \param encoding_Type : 인코딩 타입
	 */
	FileManager(string file_Name, Type data_Type = Type::Real, Type encoding_Type = Type::Categorical);
	/**
	 * 각각의 칼럼에 타입을 설정하여 데이터 생성.
	 * 
	 * \param file_Name : 파일 이름
	 * \param data_TypeList : 데이터 타입
	 * \param encoding_Type : 인코딩 타입
	 */
	FileManager(string file_Name, vector<Type> data_Types, Type encoding_Type = Type::Categorical);
	~FileManager();
	/**
	 * 해당 행렬 데이터 가져오기.
	 * 
	 * \param row : 행
	 * \param column : 열
	 * \return 데이터
	 */
	double GetData(int row, int column);
	/**
	 * 해당 행 데이터 가져오기.
	 * 
	 * \param row : 행
	 * \return 행 데이터
	 */
	vector<double> GetData(int row);
	/**
	 * 해당 칼럼의 인코딩된 데이터의 해당 행 가져오기.
	 * 
	 * \param row : 행
	 * \param column : 열
	 * \return 인코딩 데이터
	 */
	vector<double> GetEncodingData(int row, int column);
	/**
	 * 해당 칼럼들의 인코딩된 데이터의 해당 행 가져오기.
	 * 
	 * \param row : 행
	 * \param columns : 열
	 * \return 인코딩 데이터
	 */
	vector<double> GetEncodingData(int row, vector<int> columns);
	/**
	 * 표 형태로 읽기.
	 * 
	 * \param file_Name : 파일 이름
	 * \return 행열 데이터
	 */
	vector<vector<string>> GetTable(string file_Name);
	/**
	 * 행 개수 가져오기.
	 * 
	 * \return 행 개수
	 */
	int GetRowCount();


private:
	struct Data
	{
		string columnName;
		Type type = Type::Real;
		// <행, 값>
		map<int, string> value;
	};
	// <열, 값>
	map<int, Data> data;
	// 타입 리스트
	vector<Type> typeList;
	// <칼럼인덱스, <값, 인코딩>>
	map<int, map<string, double>> encodingList;
	// <칼럼인덱스, <값, 디코딩>>
	map<int, map<double, string>> decodingList;
	// 기존 칼럼과 매칭되는 인코딩 칼럼 리스트 <칼럼인덱스, 칼럼명>
	map<int, vector<int>> oneHotEncodingColumnList;

	/**
	 * 칼럼명 설정.
	 * 
	 * \param line : 첫 행
	 * \return 칼럼명
	 */
	vector<string> SetColumnName(string first_line);
	/**
	 * 모든 타입을 동일하게 설정.
	 * 
	 * \param column_Names : 칼럼명
	 * \param data_Type : 데이터 타입
	 */
	vector<Type> SetDataType(vector<string> column_Names, Type data_Type);
	/**
	 * 각 칼럼의 타입을 순서대로 설정.
	 * 
	 * \param column_Names : 칼럼명
	 * \param data_TypeList : 데이터 타입
	 */
	vector<Type> SetDataType(vector<string> column_Names, vector<Type> data_Types);
	/**
	 * 데이터 설정.
	 * 
	 * \param row : 행 인덱스
	 * \param line : 행 데이터
	 */
	map<int, string> SetData(int row, string line, int column_Count);
	/**
	 * 인코딩 리스트 추가.
	 * 
	 * \param column : 열 인덱스
	 * \param value : string 형 데이터 값
	 */
	void AddEncoingList(int column, string value);
	/**
	 * String 타입의 변수를 OneHot Encoding.
	 * 
	 * \param tmp_Data
	 * \param column_Name : 칼럼명 리스트
	 */
	void OneHotEncoding(vector<map<int, string>>& tmp_Data, vector<string>& column_Name);
	/**
	 * String 타입의 변수를 Categorical Encoding.
	 *
	 * \param tmp_Data
	 * \param column_Name : 칼럼명 리스트
	 */
	void CategoricalEncoding(vector<map<int, string>>& tmp_Data, vector<string>& column_Name);
};

