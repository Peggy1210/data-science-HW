# include <iostream>
# include <fstream>
# include <sstream>
# include <iomanip>
# include <map>
# include <vector>
# include <string>
# include <algorithm>

using namespace std;

typedef struct fpNode {
    int id;
    int frequency;
    fpNode *parent;
    fpNode *next;
    vector<fpNode *> child;

    bool operator < (fpNode const & n) const {
        return frequency < n.frequency;
    }
} fpNode;

typedef struct headerEntry {
    int id;
    int frequency;
    fpNode *head;
    
    bool operator < (headerEntry const & e) const {
        return frequency < e.frequency;
    }
} headerEntry;

map<int, int> getFreqItems(vector<vector<int>> *txs) {
    map<int, int> freqItems;
    for (vector<int> tx: *txs) {
        for (int item: tx) {
            freqItems[item]++;
        }
    }
    return freqItems;
}

void deleteUnfreqItems(map<int, int> &freqItems, double threshold) {
    vector<int> delKeys;
    for (auto item = freqItems.begin(); item != freqItems.end(); item++) {
        if (item->second < threshold) {
            delKeys.push_back(item->first);
        }
    }

    for (auto item = delKeys.begin(); item != delKeys.end(); item++) {
        freqItems.erase(*item);
    }
}

vector<vector<int>> rankFreqItems(map<int, int> &freqItems, vector<vector<int>> *txs) {    
    vector<vector<int>> rankedTx;
    for (auto tx = txs->begin(); tx != txs->end(); ) {
        for (auto item = tx->begin(); item != tx->end(); ) {
            if (freqItems.find(*item) == freqItems.end()) {
                item = tx->erase(item);
            } else {
                item++;
            }
        }
        if (tx->empty()) {
            tx = txs->erase(tx);
        } else {
            sort(tx->begin(), tx->end(), [&freqItems](int a, int b) {return freqItems[a] > freqItems[b]; });
            tx++;
        }
    }
    return *txs;
}

vector<headerEntry *> constructHeaderTable(map<int, int> &freqItems, double threshold) {
    vector<headerEntry *> headerTable;

    // Sort frequent table by frequencies
    vector<pair<int, int>> vec;
    for (auto i = freqItems.begin(); i != freqItems.end(); i++)
        vec.push_back(make_pair(i->first, i->second));
    sort(vec.begin(), vec.end(),
        [](const pair<int, int> &a, const pair<int, int> &b) { 
            if (a.first != b.first) return a.second > b.second;
            return a.first < b.first;
    });

    // Construct header table
    for (auto item = vec.begin(); item != vec.end(); item++) {      
        headerEntry *e = new headerEntry;
        e->id = item->first;
        e->frequency = item->second;
        e->head = NULL;
        headerTable.push_back(e);
    }
    return headerTable;
}

fpNode *constructFpTree(vector<headerEntry *> *headerTable, vector<vector<int>> *txs) {
    fpNode *root = new fpNode;
    root->id = -1;
    root->frequency = 0;
    root->parent = NULL;
    root->next = NULL;
    root->child.clear();

    for (vector<int> tx: *txs) {
        fpNode *cur = root;
        for (int item: tx) {
            bool found = false;
            for (fpNode *child: cur->child) {
                if (child->id == item) {
                    found = true;
                    child->frequency += 1;
                    cur = child;
                    break;
                }
            }
            if (!found) {
                fpNode *tmp = new fpNode;
                tmp->id = item;
                tmp->frequency = 1;
                tmp->parent = cur;
                tmp->next = NULL;
                tmp->child.clear();

                cur->child.push_back(tmp);
                cur = tmp;
                for (headerEntry *e: *headerTable) {
                    if (e->id == item) {
                        if (e->head == NULL) {
                            e->head = cur;
                        } else {
                            tmp = e->head;
                            while (tmp->next != NULL) {
                                tmp = tmp->next;
                            }
                            tmp->next = cur;
                        }
                        break;
                    }
                }
            }
        }
    }
    return root;
}

vector<vector<int>> *constructCondTree(fpNode *root) {
    vector<vector<int>> *txs = new vector<vector<int>>();
    while (root != NULL) {
        fpNode *cur = root;
        vector<int> tx;
        while (cur->parent->id != -1) {
            tx.push_back(cur->parent->id);
            cur = cur->parent;
        }
        if (tx.size() == 0) {
            root = root->next;
            continue;
        }
        for (int i = 0; i < root->frequency; i++) {
            txs->push_back(tx);
        }
        root = root->next;
    }
    return txs;
}

void findFreqPatterns(vector<headerEntry *> *headerTable, fpNode *root, vector<vector<int>> *freqPatterns, vector<int> *prefix, double threshold) {
    reverse(headerTable->begin(), headerTable->end());

    for (headerEntry *e: *headerTable) {
        // Generate a new pattern from prefix
        vector<int> pattern;
        for (int p: *prefix) pattern.push_back(p);
        pattern.push_back(e->id);
        sort(pattern.begin(), pattern.end());
        freqPatterns->push_back(pattern);
        
        vector<vector<int>> *txs = constructCondTree(e->head);
        map<int, int> freqItems = getFreqItems(txs);
        deleteUnfreqItems(freqItems, threshold);
        vector<headerEntry *> newHeaderTable = constructHeaderTable(freqItems, threshold);
        if (newHeaderTable.size() != 0) {
            fpNode *condTree = constructFpTree(&newHeaderTable, txs);
            findFreqPatterns(&newHeaderTable, condTree, freqPatterns, &pattern, threshold);
        }
    }
}

/*
 * argv: minimum support, input file directory and output file directory
 */
int main (int argc, char *argv[]) {
    double minSup = stod(argv[1]);
    string inDir = argv[2];
    string outDir = argv[3];
    ifstream inFile(inDir);
    ofstream outFile;
    vector<vector<int>> txs;
    vector<vector<int>> data;

    // Get input file
    string line;
    int txCnt = 0;
    while(getline(inFile, line)) {
        stringstream ss(line);
        vector<int> tx;
        for(int item; ss >> item;) {
            tx.push_back(item);
            if(ss.peek() == ',') ss.ignore();
        }
        txs.push_back(tx);
        data.push_back(tx);
        txCnt++;
    }
    inFile.close();

    double threshold = txCnt * minSup;

    map<int, int> freqItems = getFreqItems(&txs);
    deleteUnfreqItems(freqItems, threshold);
    
    vector<vector<int>> rankedTxs = rankFreqItems(freqItems, &txs);
    vector<headerEntry *> headerTable = constructHeaderTable(freqItems, threshold);

    fpNode *fpTree = constructFpTree(&headerTable, &txs);
    
    vector<vector<int>> freqPatterns;
    vector<int> prefix;
    findFreqPatterns(&headerTable, fpTree, &freqPatterns, &prefix, threshold);

    // sort(freqPatterns.begin(), freqPatterns.end(), [](const vector<int> & a, const vector<int> & b){ 
    //     if (a.size() != b.size()) {
    //         return a.size() < b.size(); 
    //     }
    //     int i = 0;
    //     for (i = 0; i < a.size(); ++i) {
    //         if (a[i] != b[i]) 
    //             break;
                
    //     }
    //     return a[i] < b[i];
    // });

    // for (auto tx = data.begin(); tx != data.end(); ++tx) {
    //     sort(tx->begin(), tx->end());
    // }

    outFile.open(outDir);
    for (vector<int> items: freqPatterns) {
        double total = 0;
        for (vector<int> tx: data) {
            int j = 0;
            for (int i = 0; i < tx.size(); i++) {
                if (tx[i] == items[j]) {
                    j++;
                }
                if (j == items.size()) {
                    total++;
                    break;
                }
            }
        }

        double sup = total / txCnt;
        for (int j = 0; j < items.size(); ++j) {
            if (j == items.size()-1) outFile << items[j];
            else outFile << items[j] << ",";
        }
        outFile << ":" << fixed << setprecision(4) << sup;
        outFile << endl;
    }
    outFile.close();

    return 0;
}