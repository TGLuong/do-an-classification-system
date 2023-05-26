import os
import re
import pickle
import numpy as np
import stat

def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_int_csv(path, obj):
    np.savetxt(path, obj, fmt='%d', delimiter=',')


def save_float_csv(path, obj):
    np.savetxt(path, obj, fmt='%.6f', delimiter=',')

def save_string_csv(path, obj):
    np.savetxt(path, obj, fmt='%s', delimiter=',')


def load_int_csv(path):
    return np.loadtxt(path, dtype=int, delimiter=',')


def load_float_csv(path):
    return np.loadtxt(path, dtype=float, delimiter=',')

def load_string_csv(path):
    return np.loadtxt(path, dtype=str, delimiter=',')

def save_uploaded_file(file):
    if not os.path.exists("files"):
        os.makedirs("files")
    with open(os.path.join("files", 'app.apk'), "wb") as f:
        f.write(file.getbuffer())
    return "files/app.apk"

def decompile(path):
    if not os.path.exists('./decompile_folder/'):
        os.makedirs('./decompile_folder/')
    try:
        filename = os.path.basename(path).replace('.apk', '.')
        os.system("apktool d -r {} -o ./decompile_folder/{} -f".format(path, filename))
        if filename.endswith('.xapk'):
            for root, _, filenames in os.walk('./decompile_folder/' + filename):
                for subfile in filenames:
                    if subfile.endswith('.apk') and not subfile.startswith('config'):
                        print(os.path.join(root, subfile))
                        os.system(f"apktool d {os.path.join(root, subfile)} -o {os.path.join(root, subfile).replace('.apk', '.')} -f")
        return './decompile_folder/' + filename
    except:
        return 'err'



class API:
    def __init__(self, a):
        try:
            self.invoke = a[:a.find(' ')]
            api = a[a.index('}, ') + 3:a.index('->') + 2]
            for i in range(a.index(';->') + 3, len(a)):
                if not (a[i].isalpha() or a[i] == '<' or a[i] == '>'):
                    break
                api = api + a[i]
            self.api = api
            pass
        except:
            self.api = ''
            self.invoke = ''


def list_smali_file(path):
    list_smali = []
    for root, _, filenames in os.walk(path + '/smali'):
        for filename in filenames:
            if filename.endswith('.smali'):
                list_smali.append(os.path.join(root, filename))
    return list_smali


def extract_apis(path, api_dataset, app_name, label):
    """
    extract api of android app
    @param[in] path : path of android app
    @param[in] api_dataset : list of api is extracted in anroid app
    @param[in] label : the label is set to the file

    return 
    """
    data = dict()
    for a in api_dataset:
        data[a] = {
            "invoke": [],
            "method": []
        }
    for smaliFile in list_smali_file(path):
        try:
            file_content = open(smaliFile, 'r').readlines()
            method_name = ''
            for row in file_content:
                if re.match(r'^.method', row):
                    method_name = row[:row.find('(')]
                    continue
                if method_name != '':
                    row = row.strip()
                    if re.search(r'^invoke', row):
                        a = API(row)
                        if a.api in api_dataset:
                            if a.invoke not in data[a.api]["invoke"]:
                                data[a.api]["invoke"].append(a.invoke)
                            if method_name not in data[a.api]["method"]:
                                data[a.api]["method"].append(method_name)
                else:
                    continue
                if re.match(r'^.end method', row):
                    method_name = ''
        except:
            pass
    os.system('rmdir /s /q \"{}\"'.format(path))
    # os.system('rm -rf {}'.format(path))
    app_data = {
        "name": app_name,
        "label": label,
        "data": data
    }
    return app_data

def get_api(file, api_dataset):
    path = decompile(file)
    os.system('rmdir /s /q \"files\"')
    return extract_apis(
                path = path, 
                api_dataset = api_dataset, 
                app_name = os.path.basename(file).replace('.apk', ''), 
                label = "unknow"
            )

def get_apis(path, api_dataset):

    path_benign = path + '/Benign/'
    path_adware = path + '/Adware/'
    path_botnet = path + '/Botnet/'
    path_ransomware = path + '/Ransomware/'
    path_scareware = path + '/Scareware/'
    path_smsmalware = path + '/SMSmalware/'

    list_benign = [path_benign + f for f in os.listdir(path_benign)]
    list_adware = [path_adware + f for f in os.listdir(path_adware)]
    list_botnet = [path_botnet + f for f in os.listdir(path_botnet)]
    list_ransomware = [path_ransomware + f for f in os.listdir(path_ransomware)]
    list_scareware = [path_scareware + f for f in os.listdir(path_scareware)]
    list_smsmalware = [path_smsmalware + f for f in os.listdir(path_smsmalware)]
    
    data = []
    for i, file in enumerate(list_benign):
        os.system('cls')
        print(f'benign :{i}')
        path = decompile(file)
        if path != 'err':
            data.append(extract_apis(
                path = path, 
                api_dataset = api_dataset, 
                app_name = os.path.basename(file).replace('.apk', ''), 
                label = "benign"
            ))
    
    for i, file in enumerate(list_adware):
        os.system('cls')
        print(f'adware :{i}')
        path = decompile(file)
        if path != 'err':
            data.append(extract_apis(
                path = path, 
                api_dataset = api_dataset, 
                app_name = os.path.basename(file).replace('.apk', ''), 
                label = "adware"
            ))

    for i, file in enumerate(list_botnet):
        os.system('cls')
        print(f'botnet :{i}')
        path = decompile(file)
        if path != 'err':
            data.append(extract_apis(
                path = path, 
                api_dataset = api_dataset, 
                app_name = os.path.basename(file).replace('.apk', ''), 
                label = "botnet"
            ))

    for i, file in enumerate(list_ransomware):
        os.system('cls')
        print(f'ransomeware :{i}')
        path = decompile(file)
        if path != 'err':
            data.append(extract_apis(
                path = path, 
                api_dataset = api_dataset, 
                app_name = os.path.basename(file).replace('.apk', ''), 
                label = "ransomeware"
            ))

    for i, file in enumerate(list_scareware):
        os.system('cls')
        print(f'scareware :{i}')
        path = decompile(file)
        if path != 'err':
            data.append(extract_apis(
                path = path, 
                api_dataset = api_dataset, 
                app_name = os.path.basename(file).replace('.apk', ''), 
                label = "scareware"
            ))

    for i, file in enumerate(list_smsmalware):
        os.system('cls')
        print(f'smsmalware :{i}')
        path = decompile(file)
        if path != 'err':
            data.append(extract_apis(
                path = path, 
                api_dataset = api_dataset, 
                app_name = os.path.basename(file).replace('.apk', ''), 
                label = "smsmalware"
            ))
    
    return data


def label(extract_data):
    labels = []
    for e in extract_data:
        labels.append(e['label'])
    return np.copy(labels)


def app_api(extract_data, api_dataset):
    matrix = []
    for element in extract_data:
        row = []
        for api in api_dataset:
            if len(element["data"][api]["invoke"]) > 0:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    return np.array(matrix)


def invoke(extract_data, api_dataset):
    invoke_matrix = np.full((len(api_dataset), len(api_dataset)), 0)
    for app in extract_data:
        for api_i in range(len(api_dataset)):
            invoke_matrix[api_i][api_i] = 1
            for inv in app["data"][api_dataset[api_i]]["invoke"]:
                for api_j in range(api_i + 1, len(api_dataset)):
                    if inv in app["data"][api_dataset[api_j]]["invoke"]:
                        invoke_matrix[api_i][api_j] = 1
                        invoke_matrix[api_j][api_i] = 1
    return invoke_matrix


def package(api_dataset):
    package_matrix = np.full((len(api_dataset), len(api_dataset)), 0)

    for api_i in range(len(api_dataset)):
        package_matrix[api_i][api_i] = 1
        for api_j in range(api_i + 1, len(api_dataset)):
            if api_dataset[api_i][:api_dataset[api_i].index(';->')] == api_dataset[api_j][
                                                                       :api_dataset[api_j].index(';->')]:
                package_matrix[api_i][api_j] = 1
                package_matrix[api_j][api_i] = 1
    return package_matrix


def method(extract_data, api_dataset):
    block_matrix = np.full((len(api_dataset), len(api_dataset)), 0)
    for app in extract_data:
        for api_i in range(len(api_dataset)):
            block_matrix[api_i][api_i] = 1
            for block in app["data"][api_dataset[api_i]]["method"]:
                for api_j in range(api_i + 1, len(api_dataset)):
                    if block in app["data"][api_dataset[api_j]]["method"]:
                        block_matrix[api_i][api_j] = 1
                        block_matrix[api_j][api_i] = 1
    return block_matrix


def create_vector_with_label(list_vector, malware):
    result = np.full((len(list_vector[0]), len(list_vector) * 2), 0.0, dtype=float)
    for i in range(len(list_vector)):
        print(f'{i + 1} in process')
        vector = list_vector[i]
        for j in range(len(vector)):
            result[j][i * 2] = np.sum(vector[j][:malware])
            result[j][i * 2 + 1] = np.sum(vector[j][malware:])
        del vector
    return result


def feature_vector_v32(list_vector_path, malware):
    vector = load_pkl(list_vector_path[0])
    len_result = len(vector)
    del vector
    result = np.full((len_result, len(list_vector_path) * 2), 0.0, dtype=float)
    for i in range(len(list_vector_path)):
        print(list_vector_path[i] + ' in process')
        vector = load_pkl(list_vector_path[i])
        for j in range(len(vector)):
            result[j][i * 2] = sum(vector[j][:malware])
            result[j][i * 2 + 1] = sum(vector[j][malware:])
        del vector
    return result

def create_feature_vector_v96(list_vector_path, benign, adware, botnet, ransomeware, scareware, smsmalware):
    adware +=benign
    botnet +=adware
    ransomeware +=botnet
    scareware += ransomeware
    smsmalware +=scareware

    vector = load_float_csv(list_vector_path[0])
    result = np.full((len(vector), 96), 0.0, dtype=float)
    for i, path in enumerate(list_vector_path):
        print(f'{i + 1} in process')
        vector = load_float_csv(path)
        for j, ele in enumerate(vector):
            result[j][i * 6] = np.sum(ele[0:benign])
            result[j][i * 6 + 1] = np.sum(ele[benign:adware])
            result[j][i * 6 + 2] = np.sum(ele[adware:botnet])
            result[j][i * 6 + 3] = np.sum(ele[botnet:ransomeware])
            result[j][i * 6 + 4] = np.sum(ele[ransomeware:scareware])
            result[j][i * 6 + 5] = np.sum(ele[scareware:smsmalware])
    return result



def feature_vector_v16(list_vector_path):
    vector = load_pkl(list_vector_path[0])
    len_result = len(vector)
    del vector
    result = np.full((len_result, len(list_vector_path)), 0.0, dtype=float)
    for i in range(len(list_vector_path)):
        print(list_vector_path[i] + ' in process')
        vector = load_pkl(list_vector_path[i])
        for j in range(len(vector)):
            result[j][i] = sum(vector[j])
        del vector
    return result

def rmtree(path):
    """
    remove tree
    param[in] : path to remove
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(path)

def create_vector(avg_sim, adware, banking, benign, riskware, smsmalware):
    banking += adware
    benign += banking
    riskware += benign
    smsmalware += riskware

    result = np.full((1, 5 * 16), 0.0, dtype=float)
    meta_path = []
    meta_path.append(avg_sim.meta_path_1().tolist())
    meta_path.append(avg_sim.meta_path_2().tolist())
    meta_path.append(avg_sim.meta_path_3().tolist())
    meta_path.append(avg_sim.meta_path_4().tolist())
    meta_path.append(avg_sim.meta_path_5().tolist())
    meta_path.append(avg_sim.meta_path_6().tolist())
    meta_path.append(avg_sim.meta_path_7().tolist())
    meta_path.append(avg_sim.meta_path_8().tolist())
    meta_path.append(avg_sim.meta_path_9().tolist())
    meta_path.append(avg_sim.meta_path_10().tolist())
    meta_path.append(avg_sim.meta_path_11().tolist())
    meta_path.append(avg_sim.meta_path_12().tolist())
    meta_path.append(avg_sim.meta_path_13().tolist())
    meta_path.append(avg_sim.meta_path_14().tolist())
    meta_path.append(avg_sim.meta_path_15().tolist())
    meta_path.append(avg_sim.meta_path_16().tolist())
    for i, ele in enumerate(meta_path):
        result[0][i * 5] = np.sum(ele[0][0:adware])
        result[0][i * 5 + 1] = np.sum(ele[0][adware:banking])
        result[0][i * 5 + 2] = np.sum(ele[0][banking:benign])
        result[0][i * 5 + 3] = np.sum(ele[0][benign:riskware])
        result[0][i * 5 + 4] = np.sum(ele[0][riskware:smsmalware])
    return result
