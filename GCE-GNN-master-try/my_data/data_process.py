import pandas as pd
import argparse


def pro_df_session(item):
    item = item[1:-1].split(', ')
    for i in range(len(item)):
        item[i] = int(item[i])
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="Cell", type=str, help="Beauty/Cell")
    parser.add_argument("--add_test", default=False, action="store_true", help="Add -1 at last or not")

    args = parser.parse_args()
    file_path = 'Amazon_' + args.data_name + '/train_sessions.csv'
    df = pd.read_csv(file_path)
    df['session'] = df['session'].apply(pro_df_session)

    output = pd.DataFrame(columns=['item_ids'])
    temp_ids = []

    for i in df.index:
        temp = df.iloc[i, 0]
        temp.append(df.iloc[i, 1])

        flag = True
        for j in range(min(len(temp_ids), len(temp))):
            if temp_ids[j] != temp[j]:
                flag = False
        if flag:
            temp_ids = temp
        else:
            if args.add_test:
                temp_ids.append(-1)     # 末尾补-1
            output = pd.concat([output, pd.DataFrame({'item_ids': [str(temp_ids)]})], ignore_index=True)
            temp_ids = temp

    with open('Amazon_' + args.data_name + '/' + args.data_name + '_change.txt', 'w') as f:
        for i in output.index:
            write = str(i+1)
            for j in output.iloc[i, 0][1:-1].split(', '):
                write = write + ' ' + j
            f.write(write + '\n')


if __name__ == '__main__':
    main()