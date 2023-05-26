import numpy as np


def row(matrix):
    result_matrix = np.full(matrix.shape, 0.0)
    np.seterr(invalid='ignore')
    try:
        for i in range(len(matrix)):
            count_row = np.sum(matrix[i] == 1)
            result_matrix[i] = np.where(matrix[i] == 1, matrix[i] / count_row, matrix[i])
    except NameError:
        print(NameError)
    return result_matrix


def column(matrix):
    result_matrix = np.full(matrix.shape, 0.0)
    np.seterr(invalid='ignore')
    try:
        for j in range(len(matrix[0])):
            count_column = np.sum(matrix[:, j] == 1)
            result_matrix[:, j] = np.where(matrix[:, j] == 1, matrix[:, j] / count_column, matrix[:, j])
    except NameError:
        print(NameError)
    return result_matrix


class AvgSimForDetect:
    def __init__(self, app_api, invoke, package, method) -> None:
        self.__r_app_api = row(app_api)
        self.__r_new_app = np.matrix(self.__r_app_api[len(self.__r_app_api) - 1])
        self.__r_app_api_t = row(app_api.T)
        self.__r_invoke = row(invoke)
        self.__r_package = row(package)
        self.__r_method = row(method)

        self.__c_app_api = column(app_api)
        self.__c_new_app = np.matrix(self.__c_app_api[len(self.__c_app_api) - 1])
        self.__c_app_api_t = column(app_api.T)
        self.__c_invoke = column(invoke)
        self.__c_package = column(package)
        self.__c_method = column(method)

    def meta_path_1(self):
        return 0.5 * (self.__r_new_app @ self.__r_app_api_t + self.__c_new_app @ self.__c_app_api_t)

    def meta_path_2(self):
        return 0.5 * (self.__r_new_app @ self.__r_method @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_method @ self.__c_app_api_t)

    def meta_path_3(self):
        return 0.5 * (self.__r_new_app @ self.__r_package @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_package @ self.__c_app_api_t)

    def meta_path_4(self):
        return 0.5 * (self.__r_new_app @ self.__r_invoke @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_invoke @ self.__c_app_api_t)

    def meta_path_5(self):
        return 0.5 * (self.__r_new_app @ self.__r_method @ self.__r_package @ self.__r_method @
                      self.__r_app_api_t + self.__c_new_app @ self.__c_method @ self.__c_package @
                      self.__c_method @ self.__c_app_api_t)

    def meta_path_6(self):
        return 0.5 * (self.__r_new_app @ self.__r_package @ self.__r_method @
                      self.__r_package @ self.__r_app_api_t + self.__c_new_app @
                      self.__c_package @ self.__c_method @ self.__c_package @ self.__c_app_api_t)

    def meta_path_7(self):
        return 0.5 * (self.__r_new_app @ self.__r_method @ self.__r_invoke @
                      self.__r_method @ self.__r_app_api_t + self.__c_new_app @ self.__c_method @
                      self.__c_invoke @ self.__c_method @ self.__c_app_api_t)

    def meta_path_8(self):
        return 0.5 * (self.__r_new_app @ self.__r_invoke @ self.__r_method @
                      self.__r_invoke @ self.__r_app_api_t + self.__c_new_app @ self.__c_invoke @
                      self.__c_method @ self.__c_invoke @ self.__c_app_api_t)

    def meta_path_9(self):
        return 0.5 * (self.__r_new_app @ self.__r_package @ self.__r_invoke @
                      self.__r_package @ self.__r_app_api_t + self.__c_new_app @
                      self.__c_package @ self.__c_invoke @ self.__c_package @ self.__c_app_api_t)

    def meta_path_10(self):
        return 0.5 * (self.__r_new_app @ self.__r_invoke @ self.__r_package @
                      self.__r_invoke @ self.__r_app_api_t + self.__c_new_app @ self.__c_invoke @
                      self.__c_package @ self.__c_invoke @ self.__c_app_api_t)

    def meta_path_11(self):
        return 0.5 * (self.__r_new_app @ self.__r_method @ self.__r_package @
                      self.__r_invoke @ self.__r_package @ self.__r_method @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_method @ self.__c_package @ self.__c_invoke @
                      self.__c_package @ self.__c_method @ self.__c_app_api_t)

    def meta_path_12(self):
        return 0.5 * (self.__r_new_app @ self.__r_package @ self.__r_method @
                      self.__r_invoke @ self.__r_method @ self.__r_package @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_package @ self.__c_method @ self.__c_invoke @
                      self.__c_method @ self.__c_package @ self.__c_app_api_t)

    def meta_path_13(self):
        return 0.5 * (self.__r_new_app @ self.__r_method @ self.__r_invoke @
                      self.__r_package @ self.__r_invoke @ self.__r_method @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_method @ self.__c_invoke @ self.__c_package @
                      self.__c_invoke @ self.__c_method @ self.__c_app_api_t)

    def meta_path_14(self):
        return 0.5 * (self.__r_new_app @ self.__r_invoke @ self.__r_method @
                      self.__r_package @ self.__r_method @ self.__r_invoke @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_invoke @ self.__c_method @ self.__c_package @
                      self.__c_method @ self.__c_invoke @ self.__c_app_api_t)

    def meta_path_15(self):
        return 0.5 * (self.__r_new_app @ self.__r_invoke @ self.__r_package @
                      self.__r_method @ self.__r_package @ self.__r_invoke @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_invoke @ self.__c_package @ self.__c_method @
                      self.__c_package @ self.__c_invoke @ self.__c_app_api_t)

    def meta_path_16(self):
        return 0.5 * (self.__r_new_app @ self.__r_package @ self.__r_invoke @
                      self.__r_method @ self.__r_invoke @ self.__r_package @ self.__r_app_api_t +
                      self.__c_new_app @ self.__c_package @ self.__c_invoke @ self.__c_method @
                      self.__c_invoke @ self.__c_package @ self.__c_app_api_t)
