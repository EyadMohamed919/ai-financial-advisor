package com.aifinancialadvisor.user;

import java.util.UUID;

public class User {

    private UUID id;
    private String email;
    private String passwordHash;
    private Role role;
    private boolean verified;

    public User(String email, String passwordHash) {
        this.id = UUID.randomUUID();
        this.email = email;
        this.passwordHash = passwordHash;
        this.role = Role.USER;
        this.verified = false;
    }

    public UUID getId() {
        return id;
    }

    public String getEmail() {
        return email;
    }

    public Role getRole() {
        return role;
    }

    public boolean isVerified() {
        return verified;
    }

    public void verifyEmail() {
        this.verified = true;
    }
}