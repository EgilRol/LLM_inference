// This is .h file for tokenizer
// You find the implementation under src/tokenizer_bpe.cpp 
// You have to complete that file

#pragma once

#include "prelude.h"

class LLMTokenizer {
  public:
    virtual ~LLMTokenizer() = default;

    // "Hello world" -> 9906 1917
    virtual vector<int> encode(const string &text) const = 0;

    // 9906 1917 -> "Hello world"
    // In our project ids are always single sequence since we decode one by one
    virtual string decode(const vector<int> &ids) const = 0;

    virtual int bos_id() const = 0;
    virtual int eos_id() const = 0;
};

// *****************************************************************************

class BPETokenizer final : public LLMTokenizer {
  public:
    unordered_map<string, int> rank;
    vector<string> id2tok;

    unordered_map<string, int> special2id;
    unordered_map<int, string> id2special;
    vector<string> specials_sorted;

    int bos_id_ = -1;
    int eos_id_ = -1;

    BPETokenizer();
    BPETokenizer(const string &path);
    vector<int> encode(const string &text) const override;
    string decode(const vector<int> &ids) const override;
    int bos_id() const override;
    int eos_id() const override;

    vector<int> encode_no_merge(const string &text) const;

  private:
    static string b64decode(const string &s);

    vector<int> encode_impl(const string &text, bool enable_merge) const;

    vector<int> encode_chunk(const string &s, bool enable_merge) const;
};
