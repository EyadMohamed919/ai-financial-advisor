package com.aifinancialadvisor.user;

import org.springframework.stereotype.Service;

@Service
public class UserService {

    public User createUser(String email, String passwordHash) {
        return new User(email, passwordHash);
    }
}
// it doesnt save data yet, when database is connected, i will continue to
// implement the save functionality